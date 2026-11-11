defmodule NxStereo.FullRangeTest do
  use ExUnit.Case, async: true

  @moduletag :vulkan
  setup_all do: NxStereo.TestDevices.vulkan()

  for {height, width} <- [{19, 101}, {90, 70}] do
    test "three-path full matching preserves reference decisions at #{height}x#{width}", %{
      devices: devices
    } do
      shape = {unquote(height), unquote(width)}
      {left, _key} = Nx.Random.randint(Nx.Random.key(53), 0, 256, shape: shape, type: :s32)
      {right, _key} = Nx.Random.randint(Nx.Random.key(92), 0, 256, shape: shape, type: :s32)
      match = &NxStereo.sgm(&1, &2, disparities: 128, paths: 3, p1: &3, p2: &4)
      templates = [left, right, Nx.template({}, :f32), Nx.template({}, :f32)]
      reference = Nx.Defn.compile(match, templates, compiler: EXLA)

      for device <- devices do
        compiled =
          Nx.Defn.compile(match, templates,
            compiler: NxIREE.Compiler,
            iree_runtime_options: [device: device]
          )

        for {p1, p2} <- [{0, 1}, {8, 64}, {78, 79}] do
          inputs = [left, right, Nx.f32(p1), Nx.f32(p2)]
          expected = apply(reference, inputs) |> Nx.to_flat_list()
          actual = apply(compiled, inputs) |> Nx.to_flat_list()

          assert Enum.zip(actual, expected)
                 |> Enum.all?(fn {got, want} ->
                   got < 0 == want < 0 and abs(got - want) <= 1.0e-4
                 end),
                 "#{device.uri}, penalties #{p1}/#{p2}"
        end
      end
    end
  end
end
