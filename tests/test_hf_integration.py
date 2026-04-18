def test_hf_model_integration():
    from transformers import AutoModel, AutoTokenizer

    model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        assert tokenizer is not None
        assert model is not None
    except Exception as e:
        assert False, f"Model loading failed: {str(e)}"