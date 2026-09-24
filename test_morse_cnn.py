"""morse_cnn.py のユニットテスト(Milestone 6)。"""

import torch

from morse_cnn import DEFAULT_CHANNELS, MorseCNN


def test_forward_output_shape_matches_num_classes():
    model = MorseCNN(num_classes=62)
    x = torch.randn(4, 1, 64, 412)
    logits = model(x)
    assert logits.shape == (4, 62)


def test_forward_accepts_varying_time_dimension():
    """AdaptiveAvgPoolのため、時間方向のサイズが変わっても動作すること。"""
    model = MorseCNN(num_classes=10)
    for frames in (32, 100, 256):
        x = torch.randn(2, 1, 64, frames)
        logits = model(x)
        assert logits.shape == (2, 10)


def test_forward_accepts_batch_size_one():
    model = MorseCNN(num_classes=5)
    x = torch.randn(1, 1, 64, 100)
    logits = model(x)
    assert logits.shape == (1, 5)


def test_model_has_trainable_parameters():
    model = MorseCNN(num_classes=62)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params > 0


def test_output_is_finite():
    model = MorseCNN(num_classes=8)
    x = torch.randn(3, 1, 64, 100)
    logits = model(x)
    assert torch.isfinite(logits).all()


def test_default_channels_have_four_stages():
    assert len(DEFAULT_CHANNELS) == 4


def test_config_roundtrip_reconstructs_equivalent_architecture():
    model = MorseCNN(num_classes=10, channels=(8, 16), hidden_dim=32, dropout=0.1)
    config = model.config()
    rebuilt = MorseCNN.from_config(config)

    assert rebuilt.channels == model.channels
    assert rebuilt.hidden_dim == model.hidden_dim
    assert rebuilt.num_classes == model.num_classes

    x = torch.randn(2, 1, 64, 100)
    assert rebuilt(x).shape == model(x).shape


def test_from_config_falls_back_to_defaults_for_missing_keys():
    """旧バージョンのチェックポイント(num_classesしか無い)からでも再構築できる。"""
    model = MorseCNN.from_config({"num_classes": 62})
    assert model.channels == DEFAULT_CHANNELS
    x = torch.randn(1, 1, 64, 100)
    assert model(x).shape == (1, 62)


def test_custom_channels_produce_matching_final_feature_size():
    model = MorseCNN(num_classes=5, channels=(4, 8, 16))
    x = torch.randn(1, 1, 64, 50)
    assert model(x).shape == (1, 5)


# --- MorseCRNN(畳み込み + 双方向GRU) -------------------------------------


def test_crnn_forward_output_shape_and_varying_time():
    from morse_cnn import MorseCRNN

    model = MorseCRNN(num_classes=62).eval()
    assert model(torch.randn(3, 1, 64, 412)).shape == (3, 62)
    for frames in (32, 100, 256):
        assert model(torch.randn(2, 1, 64, frames)).shape == (2, 62)
    assert torch.isfinite(model(torch.randn(2, 1, 64, 100))).all()


def test_crnn_config_roundtrip_and_arch_dispatch():
    from morse_cnn import MorseCRNN, MorseCNN, create_model, model_from_config

    model = MorseCRNN(num_classes=7, channels=(8, 16), rnn_hidden=16, hidden_dim=32, dropout=0.2)
    config = model.config()
    assert config["arch"] == "crnn"

    rebuilt = model_from_config(config)
    assert isinstance(rebuilt, MorseCRNN)
    rebuilt.load_state_dict(model.state_dict())  # 形状が一致すること
    x = torch.randn(2, 1, 64, 60)
    assert rebuilt.eval()(x).shape == (2, 7)

    assert isinstance(create_model("cnn", 5), MorseCNN)
    assert isinstance(create_model("crnn", 5), MorseCRNN)


def test_model_from_config_treats_missing_arch_as_cnn():
    """archキーの無い旧チェックポイントはMorseCNNとして再構築される(後方互換)。"""
    from morse_cnn import MorseCNN, model_from_config

    assert isinstance(model_from_config({"num_classes": 62}), MorseCNN)


def test_create_model_rejects_unknown_arch():
    import pytest

    from morse_cnn import create_model, model_from_config

    with pytest.raises(ValueError):
        create_model("transformer", 5)
    with pytest.raises(ValueError):
        model_from_config({"arch": "transformer", "num_classes": 5})


def test_crnn_output_depends_on_temporal_order():
    """時間方向の並びを入れ替えると出力が変わる(順序を扱えるモデルであること)。"""
    from morse_cnn import MorseCRNN

    torch.manual_seed(0)
    model = MorseCRNN(num_classes=10, channels=(8, 16, 16, 16), rnn_hidden=16).eval()
    x = torch.randn(1, 1, 64, 120)
    flipped = torch.flip(x, dims=[3])
    assert not torch.allclose(model(x), model(flipped), atol=1e-4)
