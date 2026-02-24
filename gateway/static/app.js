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
    alert(data.message);
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
clearBtn.addEventListener("click", (event) => {
    event.stopPropagation();
    fileInput.value = "";
    updateSelectedFile(null);
});

browseBtn.addEventListener("click", (event) => {
    event.stopPropagation();
    fileInput.click();
});
