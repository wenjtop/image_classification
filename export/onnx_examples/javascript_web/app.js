const fileInput = document.getElementById('fileInput');
const classifyButton = document.getElementById('classifyButton');
const resultDiv = document.getElementById('result');

let ortSession = null;
async function loadModel() {
    ortSession = await ort.InferenceSession.create('mobilenet_v3_s.onnx');
    console.log('Model loaded');
}

async function classifyImage() {
    if (!ortSession) {
        resultDiv.innerText = 'Model not loaded.';
        return;
    }

    const file = fileInput.files[0];
    if (!file) {
        resultDiv.innerText = 'No image selected.';
        return;
    }

    const img = new Image();
    img.src = URL.createObjectURL(file);
    img.onload = async () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);

        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const input = preprocessImage(imageData);

        const tensor = new ort.Tensor('float32', input, [1, 3, canvas.height, canvas.width]);
        const feeds = { input: tensor };

        try {
            const results = await ortSession.run(feeds);
            const output = results.output.data;
            resultDiv.innerText = `Prediction: ${getPrediction(output)}`;
        } catch (error) {
            console.error('Error during inference:', error);
            resultDiv.innerText = 'Error during inference.';
        }
    };
}

function preprocessImage(imageData) {
    const { data, width, height } = imageData;
    const input = new Float32Array(3 * width * height);

    for (let i = 0; i < width * height; i++) {
        const r = data[i * 4] / 255;
        const g = data[i * 4 + 1] / 255;
        const b = data[i * 4 + 2] / 255;

        input[i] = r;
        input[i + width * height] = g;
        input[i + 2 * width * height] = b;
    }

    return input;
}

function getPrediction(output) {
    // This is a placeholder. Adjust based on your model's output format.
    // For a binary classification, you might return the index of the highest probability.
    return output;
}

classifyButton.addEventListener('click', classifyImage);
loadModel();
