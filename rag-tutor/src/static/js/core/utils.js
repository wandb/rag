// Utility function to scroll elements to bottom
export function scrollToBottom(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        requestAnimationFrame(() => {
            element.scrollTop = element.scrollHeight;
        });
    }
}

// Helper function to create WAV header
export function createWavHeader(dataLength) {
    const OPENAI_SAMPLE_RATE = 24000;  // OpenAI uses 24kHz
    const header = new ArrayBuffer(44);
    const view = new DataView(header);

    const totalLength = Array.isArray(dataLength)
        ? dataLength.reduce((acc, chunk) => acc + chunk.length, 0)
        : dataLength;

    // "RIFF" chunk descriptor
    view.setUint32(0, 0x52494646, false); // "RIFF"
    view.setUint32(4, 36 + totalLength, true); // File size
    view.setUint32(8, 0x57415645, false); // "WAVE"

    // "fmt " sub-chunk
    view.setUint32(12, 0x666D7420, false); // "fmt "
    view.setUint32(16, 16, true); // Subchunk1Size (16 for PCM)
    view.setUint16(20, 1, true); // AudioFormat (1 for PCM)
    view.setUint16(22, 1, true); // NumChannels (1 for mono)
    view.setUint32(24, OPENAI_SAMPLE_RATE, true); // SampleRate
    view.setUint32(28, OPENAI_SAMPLE_RATE * 2, true); // ByteRate
    view.setUint16(32, 2, true); // BlockAlign
    view.setUint16(34, 16, true); // BitsPerSample

    // "data" sub-chunk
    view.setUint32(36, 0x64617461, false); // "data"
    view.setUint32(40, totalLength, true); // Subchunk2Size

    return new Uint8Array(header);
} 