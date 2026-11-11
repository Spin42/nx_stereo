defmodule NxStereo.WideBandTest do
  use ExUnit.Case, async: true

  @moduletag :vulkan
  setup_all do: NxStereo.TestDevices.vulkan()

  for band <- [32, 64] do
    test "band #{band} preserves reference decisions at penalty limits", %{devices: devices} do
      shape = {18, 98}
      {left, _key} = Nx.Random.randint(Nx.Random.key(53), 0, 256, shape: shape, type: :s32)
      {right, _key} = Nx.Random.randint(Nx.Random.key(92), 0, 256, shape: shape, type: :s32)
      phase = Nx.add(Nx.iota(shape, axis: 1), Nx.multiply(Nx.iota(shape, axis: 0), 2))
      base = Nx.select(Nx.equal(Nx.remainder(phase, 3), 0), 0, 128 - unquote(band))

      match =
        &NxStereo.sgm_band(&1, &2, &3,
          disparities: 128,
          band: unquote(band),
          paths: 8,
          p1: &4,
          p2: &5
        )

      templates = [left, right, base, Nx.template({}, :f32), Nx.template({}, :f32)]
      reference = Nx.Defn.compile(match, templates, compiler: EXLA)

      for device <- devices do
        compiled =
          Nx.Defn.compile(match, templates,
            compiler: NxIREE.Compiler,
            iree_runtime_options: [device: device]
          )

        for {p1, p2} <- [{0, 1}, {0, 79}, {78, 79}] do
          inputs = [left, right, base, Nx.f32(p1), Nx.f32(p2)]
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
