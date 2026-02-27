const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const fileName = document.getElementById("file-name");
const sendBtn = document.getElementById("send-btn");
const clearBtn = document.getElementById("clear-btn");
const browseBtn = document.getElementById("browse-btn");
let selectedFile = null;

async function showAlert() {
    if (!selectedFile) {
        alert("Please select a file first.");
        return;
    }

    const requestId = (crypto.randomUUID && crypto.randomUUID())
        || Array.from(crypto.getRandomValues(new Uint8Array(16)))
            .map((b, i) => (i === 6 ? (b & 0x0f) | 0x40 : i === 8 ? (b & 0x3f) | 0x80 : b)
                .toString(16)
                .padStart(2, "0"))
            .join("")
            .replace(/^(.{8})(.{4})(.{4})(.{4})(.{12}).*$/, "$1-$2-$3-$4-$5");
    const formData = new FormData();
    formData.append("request_id", requestId);
    formData.append("file", selectedFile);

    const response = await fetch("/api/alert", {
        method: "POST",
        body: formData,
    });

    if (!response.ok) {
        alert("Request failed.");
        return;
    }

    const data = await response.json();
    if (data.status !== 200) {
        alert("Server error: " + data.message);
        return;
    }

    await pollForResult(requestId);
}

function updateSelectedFile(file) {
    selectedFile = file;
    if (file) {
        fileName.textContent = file.name;
        dropzone.classList.add("filled");
        sendBtn.disabled = false;
    } else {
        fileName.textContent = "No file selected";
        dropzone.classList.remove("filled");
        sendBtn.disabled = true;
    }
}

fileInput.addEventListener("change", (event) => {
    const file = event.target.files && event.target.files[0];
    updateSelectedFile(file || null);
});

dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
});

dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
    const file = event.dataTransfer.files && event.dataTransfer.files[0];
    updateSelectedFile(file || null);
});

sendBtn.addEventListener("click", showAlert);

async function pollForResult(requestId) {
    const statusUrl = `/api/status/${requestId}`;
    const resultUrl = `/api/result/${requestId}`;

    while (true) {
        const res = await fetch(statusUrl);
        if (res.status === 200) {
            await new Promise((resolve) => setTimeout(resolve, 1000));
            continue;
        }

        if (res.status === 201) {
            const fileRes = await fetch(resultUrl);
            if (!fileRes.ok) {
                alert("Failed to download result.");
                return;
            }
            const blob = await fileRes.blob();
            const url = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.href = url;
            const headerFilename = fileRes.headers.get("x-result-filename");
            const contentDisposition = fileRes.headers.get("content-disposition");
            let filename = headerFilename || `${requestId}.wav`;
            if (!headerFilename && contentDisposition) {
                const match = contentDisposition.match(/filename="?([^"]+)"?/);
                if (match) filename = match[1];
            }
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            link.remove();
            URL.revokeObjectURL(url);
            return;
        }

        let errorMsg = "Processing failed.";
        try {
            const errorData = await res.json();
            if (errorData.message) {
                errorMsg += " Server message: " + errorData.message;
            }
        } catch (e) {
                // Ignore JSON parsing errors
        }

        alert(errorMsg);
        return;
    }
}
clearBtn.addEventListener("click", (event) => {
    event.stopPropagation();
    fileInput.value = "";
    updateSelectedFile(null);
});

browseBtn.addEventListener("click", (event) => {
    event.stopPropagation();
    fileInput.click();
});
