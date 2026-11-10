defmodule NxStereo.MatcherTest do
  # Each Port here is a shell script standing in for stereo_match: requests
  # and replies are 4-byte length-prefixed packets, and a reply's first byte
  # says what it is (7 configured, 8 outputs, 9 refused).
  use ExUnit.Case, async: true

  import ExUnit.CaptureLog

  alias NxStereo.Matcher

  @moduletag :capture_log
  @moduletag :tmp_dir

  defp script(dir, body) do
    path = Path.join(dir, "stereo_match")
    File.write!(path, "#!/bin/sh\n" <> body <> "\n")
    File.chmod!(path, 0o755)
    path
  end

  # Answers each request with itself.
  defp echo(dir), do: script(dir, "exec cat")

  defp start(opts),
    do: start_supervised!({Matcher, Keyword.put(opts, :name, nil)}, id: make_ref())

  defp run(matcher, version, configuration, request, owner \\ self()) do
    matcher
    |> Matcher.send_run(owner, 1, version, configuration, request)
    |> :gen_server.receive_response(5_000)
  end

  test "configures a segment on its first run and once its version changes", %{tmp_dir: dir} do
    matcher = start(path: echo(dir))

    assert run(matcher, 0, <<7, "configuration">>, <<8, "one">>) == {:reply, {:ok, "one"}}
    assert run(matcher, 0, <<9, "not sent">>, <<8, "two">>) == {:reply, {:ok, "two"}}

    assert run(matcher, 1, <<9, "refused">>, <<8, "three">>) ==
             {:reply, {:error, {:configuration_refused, "refused"}}}
  end

  test "answers requests sent without waiting for the previous reply, in the order sent",
       %{tmp_dir: dir} do
    matcher = start(path: echo(dir))

    sent =
      for word <- ["a", "b", "c"],
          do: Matcher.send_run(matcher, self(), 1, 0, <<7>>, <<8, word::binary>>)

    assert Enum.map(sent, &:gen_server.receive_response(&1, 5_000)) ==
             [{:reply, {:ok, "a"}}, {:reply, {:ok, "b"}}, {:reply, {:ok, "c"}}]
  end

  test "refuses a run the Port does not answer with outputs, logging why", %{tmp_dir: dir} do
    matcher = start(path: echo(dir))

    log =
      capture_log(fn ->
        assert run(matcher, 0, <<7>>, <<9, "no such segment">>) ==
                 {:reply, {:error, :bad_request}}
      end)

    assert log =~ "no such segment"
  end

  test "forgets a segment once the process that ran it exits", %{tmp_dir: dir} do
    matcher = start(path: echo(dir))

    owner =
      spawn(fn ->
        receive do
          :exit -> :ok
        end
      end)

    assert run(matcher, 0, <<7>>, <<8, "one">>, owner) == {:reply, {:ok, "one"}}
    assert map_size(:sys.get_state(matcher).configured) == 1

    send(owner, :exit)
    forgotten? = fn -> map_size(:sys.get_state(matcher).configured) == 0 end
    assert Enum.any?(1..100, fn _try -> forgotten?.() or (Process.sleep(10) && false) end)
  end

  test "a request without a reply in time closes the Port, failing every request waiting",
       %{tmp_dir: dir} do
    matcher = start(path: script(dir, "exec cat > /dev/null"), timeout_ms: 50)
    sent = for _ <- 1..2, do: Matcher.send_run(matcher, self(), 1, 0, <<7>>, <<8>>)

    assert Enum.map(sent, &:gen_server.receive_response(&1, 5_000)) ==
             [{:reply, {:error, :timeout}}, {:reply, {:error, :timeout}}]

    assert :sys.get_state(matcher).port == nil
  end

  test "a Port that exits fails every request waiting for it, and reopens on the next one",
       %{tmp_dir: dir} do
    answer = Path.join(dir, "answer")

    path =
      script(dir, """
      if [ -e "#{answer}" ]; then exec cat; fi
      # The configuration and both runs, 5 bytes each.
      head -c 15 > /dev/null
      exit 3
      """)

    matcher = start(path: path)
    sent = for _ <- 1..2, do: Matcher.send_run(matcher, self(), 1, 0, <<7>>, <<8>>)

    assert Enum.map(sent, &:gen_server.receive_response(&1, 5_000)) ==
             [{:reply, {:error, {:exited, 3}}}, {:reply, {:error, {:exited, 3}}}]

    File.touch!(answer)
    assert run(matcher, 0, <<7>>, <<8, "back">>) == {:reply, {:ok, "back"}}
  end

  test "without the binary, says where it looked" do
    assert run(start(path: "/nonexistent/stereo_match"), 0, <<7>>, <<8>>) ==
             {:reply, {:error, {:no_matcher, "/nonexistent/stereo_match"}}}
  end

  test "passes a positive thread count, and leaves another out with a warning" do
    assert Matcher.port_arguments(nil) == []
    assert Matcher.port_arguments(3) == ["--threads", "3"]

    log = capture_log(fn -> assert Matcher.port_arguments(0) == [] end)
    assert log =~ ":threads must be a positive integer, got 0"
  end

  test "starts the Port with the thread count of its options", %{tmp_dir: dir} do
    # Configures, then answers the run with its second argument, the thread count.
    replies = ~S(printf '\000\000\000\001\007\000\000\000\002\010'"$2")
    matcher = start(path: script(dir, replies <> "\nexec cat > /dev/null"), threads: 2)

    assert run(matcher, 0, <<7>>, <<8, "x">>) == {:reply, {:ok, "2"}}
  end
end
