defmodule NxStereo.Stage.DepthTest do
  use ExUnit.Case, async: true

  alias NxStereo.Stage.Depth

  @geometry %{focal_length_px: 400.0, baseline_m: 0.06}

  test "adds the depth of the disparity in millimetres and metres" do
    {:ok, state} = Depth.init([], @geometry)
    params = Depth.params(state, @geometry)
    tensors = Depth.step(%{disparity: Nx.tensor([[10.0, -1.0]])}, params, state)

    assert Nx.to_list(tensors.depth_mm) == [[2400, 0]]
    assert Nx.to_list(tensors.depth_m) == [[2.4000000953674316, 0.0]]
    assert Depth.device_only() == [:depth_m]
  end

  test "takes no settings" do
    assert {:error, message} = Depth.init([scale: 2], @geometry)
    assert message =~ "no settings"
  end
end
