# Research prototype for the fixed Bench.Scene chain only.
# It drops camera frames after rectification; it is not a general Stage compiler.
Code.eval_string(
  File.read!("lib/nx_stereo/nx_segment.ex")
  |> String.replace("defmodule NxStereo.NxSegment do", "defmodule SplitSegmentHelpers do")
  |> String.replace("defp bundle(", "def bundle(")
)

defmodule SplitSegment do
  @moduledoc false
  def replace(segment, cuts) do
    entries =
      Map.new(segment.specs, fn {key, {shape, type}} -> {key, Nx.template(shape, type)} end)

    params = Nx.to_template(segment.params)
    members = Enum.with_index(segment.members)
    boundaries = [0 | cuts] ++ [length(members)]
    parts = Enum.zip(boundaries, tl(boundaries))

    {compiled, _template} =
      Enum.map_reduce(parts, entries, fn {first, last}, template ->
        group = Enum.slice(members, first, last - first)
        final? = last == length(members)

        fun = fn tensors, params ->
          outputs =
            Enum.reduce(group, tensors, fn {member, index}, tensors ->
              tensors
              |> member.module.step(Map.fetch!(params, index), member.state)
              |> Map.new(fn
                {{:carry, name}, tensor} -> {{:carry, index, name}, tensor}
                entry -> entry
              end)
            end)
            |> Map.drop([:left_frame, :right_frame])

          if final?, do: SplitSegmentHelpers.bundle(outputs, segment.layout), else: outputs
        end

        output_template = Nx.Defn.debug_expr(fun).(template, params) |> Nx.to_template()

        if root = System.get_env("SPLIT_DUMP_ROOT") do
          shape = segment.geometry
          dir = "#{root}/#{shape.height}x#{shape.width}/#{Enum.join(cuts, "-")}/#{first}-#{last}"

          flags =
            Keyword.get(segment.defn_options, :iree_compiler_flags, [
              "--iree-hal-target-backends=vulkan-spirv"
            ])

          options =
            segment.defn_options
            |> Keyword.delete(:compiler)
            |> Keyword.put(:iree_compiler_flags, flags ++ ["--dump-compilation-phases-to=#{dir}"])

          {:ok, _module} = NxIREE.Compiler.to_bytecode(fun, [template, params], options)
          IO.puts("allocation inspection: #{dir}")
        end

        {Nx.Defn.compile(fun, [template, params], segment.defn_options), output_template}
      end)

    function = fn entry, params ->
      Enum.reduce(compiled, entry, fn part, values -> part.(values, params) end)
    end

    %{segment | function: function}
  end
end
