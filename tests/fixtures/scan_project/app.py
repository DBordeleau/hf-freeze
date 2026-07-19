MODEL_ID = "org/model"

model = AutoModel.from_pretrained(MODEL_ID, revision="main")  # noqa: F821
data = load_dataset("org/data")  # noqa: F821
