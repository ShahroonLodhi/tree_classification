const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

// Start webcam
navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
    video.srcObject = stream;
    video.play();
});

// Auto draw frame + send every second
setInterval(() => {
    if (video.readyState === 4) {
        captureFrameAndPredict();
    }
}, 1000);

function captureFrameAndPredict() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    const tempCtx = tempCanvas.getContext("2d");
    tempCtx.drawImage(video, 0, 0);

    tempCanvas.toBlob(blob => {
        const formData = new FormData();
        formData.append("file", blob, "frame.jpg");

        fetch("https://your-app-name.onrender.com/predict/", {
            method: "POST",
            body: formData
        })
        .then(res => res.json())
        .then(data => {
            // Redraw webcam feed
            ctx.drawImage(video, 0, 0);

            if (data.box) {
                const { x, y, w, h } = data.box;
                const label = data.label;

                // Draw bounding box
                ctx.strokeStyle = "lime";
                ctx.lineWidth = 3;
                ctx.strokeRect(x, y, w, h);

                // Draw label
                ctx.fillStyle = "black";
                ctx.font = "18px Arial";
                ctx.fillText(label, x + 5, y - 10);
            }
        })
        .catch(err => {
            console.error("Prediction error:", err);
        });
    }, "image/jpeg");
}
