# OCR: Image to Text
> Converts Image to OCR with multilingual support, returning coordinates of text in the image. The project is designed to run on CUDA if `gpu` is available, and it falls back to `mps` if CUDA is not available, or to `cpu` if neither CUDA nor MPS are available.

```python
if torch.cuda.is_available():
    self.device = 'cuda'
elif torch.backends.mps.is_available():
    self.device = 'mps'
else:
    self.device = 'cpu'
```

## Setup

To set up, run the following commands:

```bash
pip install -r requirements.txt
python app.py
```

## Gradio

Gradio is used to run the application, and it can be accessed at:

```bash
http://127.0.0.1:7860
```
Make sure you have the necessary dependencies installed before running the application.

Feel free to contribute and provide feedback to help improve this OCR project.
