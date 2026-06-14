import gradio as gr


def predict(image):
    return {"Healthy": 1.0}


interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=1),
    title="LeafLens",
    description="Upload a leaf image to get a plant health prediction.",
)


if __name__ == "__main__":
    interface.launch()
