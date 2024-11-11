import { audioState } from './state.js';

let clientCanvas, serverCanvas;
let clientCtx, serverCtx;
const visualizationIntervals = new Map();

function drawSineWave(canvas, ctx, values, color) {
    const width = canvas.width;
    const height = canvas.height;

    ctx.fillStyle = '#f8f8f8';
    ctx.fillRect(0, 0, width, height);

    ctx.beginPath();
    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 1;
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;

    const now = Date.now() / 1000;
    const points = new Float32Array(values.length);

    for (let i = 0; i < values.length; i++) {
        const oscillation = Math.sin(now * 4 + i * 0.1) * 0.15;
        points[i] = values[i] + oscillation;
    }

    ctx.moveTo(0, height / 2);

    for (let i = 0; i < width; i += 2) {
        const valueIndex = Math.floor((i / width) * points.length);
        const value = points[valueIndex] || 0;
        const nextValue = points[Math.min(valueIndex + 1, points.length - 1)] || 0;

        const normalizedValue = (value - 0.5) * 0.8;
        const normalizedNextValue = (nextValue - 0.5) * 0.8;

        const x = i;
        const nextX = Math.min(i + 2, width);
        const y = (height / 2) + (normalizedValue * height / 2);
        const nextY = (height / 2) + (normalizedNextValue * height / 2);

        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            const controlX = (x + nextX) / 2;
            const controlY = (y + nextY) / 2;
            ctx.quadraticCurveTo(x, y, controlX, controlY);
        }
    }

    ctx.stroke();
}

export function setupVisualization() {
    clientCanvas = document.getElementById('client-canvas');
    serverCanvas = document.getElementById('server-canvas');

    if (clientCanvas && serverCanvas) {
        clientCtx = clientCanvas.getContext('2d');
        serverCtx = serverCanvas.getContext('2d');

        function resizeCanvas(canvas) {
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width;
            canvas.height = rect.height;
        }

        resizeCanvas(clientCanvas);
        resizeCanvas(serverCanvas);

        window.addEventListener('resize', () => {
            resizeCanvas(clientCanvas);
            resizeCanvas(serverCanvas);
        });

        function render() {
            if (!clientCanvas || !serverCanvas) return;

            // Client-side visualization (recording)
            if (audioState.recorder && audioState.recorder.processor) {
                try {
                    const result = audioState.recorder.getFrequencies('voice');
                    drawSineWave(clientCanvas, clientCtx, audioState.isRecording ? result.values : new Float32Array(128).fill(0.5), '#0099ff');
                } catch (error) {
                    drawSineWave(clientCanvas, clientCtx, new Float32Array(128).fill(0.5), '#0099ff');
                }
            } else {
                drawSineWave(clientCanvas, clientCtx, new Float32Array(128).fill(0.5), '#0099ff');
            }

            // Server-side visualization (playback)
            if (audioState.streamPlayer && audioState.streamPlayer.analyser && audioState.isPlaying) {
                try {
                    const result = audioState.streamPlayer.getFrequencies('voice');
                    const values = result.values.map(v =>
                        v === 0.5 ? 0.5 + (Math.random() * 0.01 - 0.005) : v
                    );
                    drawSineWave(serverCanvas, serverCtx, values, '#009900');
                } catch (error) {
                    console.error('Visualization error:', error);
                    drawSineWave(serverCanvas, serverCtx, new Float32Array(128).fill(0.5), '#009900');
                }
            } else {
                drawSineWave(serverCanvas, serverCtx, new Float32Array(128).fill(0.5), '#009900');
            }

            requestAnimationFrame(render);
        }

        render();
    }
}

export function clearVisualizations() {
    if (clientCtx && clientCanvas) {
        clientCtx.clearRect(0, 0, clientCanvas.width, clientCanvas.height);
    }
    if (serverCtx && serverCanvas) {
        serverCtx.clearRect(0, 0, serverCanvas.width, serverCanvas.height);
    }
} 