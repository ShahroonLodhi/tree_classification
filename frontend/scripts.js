const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const resultLabel = document.getElementById("result");

// Start webcam
navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
    video.srcObject = stream;
});

// Automatically capture frame every second
setInterval(() => {
    if (video.readyState === 4) {
        captureAndSendFrame();
    }
}, 1000); // 1000 ms = 1 frame/sec

function captureAndSendFrame() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    canvas.toBlob(blob => {
        const formData = new FormData();
        formData.append("file", blob, "frame.jpg");

        fetch("https://your-app-name.onrender.com/predict/", {
            method: "POST",
            body: formData
        })
        .then(res => res.json())
        .then(data => {
            // Display predicted class
            resultLabel.innerText = "Predicted Class: " + data.label;

            // Display returned image (with bounding box)
            const img = new Image();
            img.src = "data:image/jpeg;base64," + data.image_base64;
            img.onload = () => ctx.drawImage(img, 0, 0);
        })
        .catch(err => {
            console.error("Error:", err);
        });
    }, "image/jpeg");
}
