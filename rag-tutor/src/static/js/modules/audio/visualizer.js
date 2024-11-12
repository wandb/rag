import { audioState } from '../../core/state.js';

let clientCanvas, serverCanvas;
let clientCtx, serverCtx;

// Add WeakMap for memoization
const dataMap = new WeakMap();

// Add normalizeArray function
function normalizeArray(data, pointCount, downsamplePeaks = true, memoize = true) {
    let cache, mKey, dKey;
    if (memoize) {
        mKey = pointCount.toString();
        dKey = downsamplePeaks.toString();
        cache = dataMap.has(data) ? dataMap.get(data) : {};
        dataMap.set(data, cache);
        cache[mKey] = cache[mKey] || {};
        if (cache[mKey][dKey]) {
            return cache[mKey][dKey];
        }
    }

    const n = data.length;
    const result = new Array(pointCount);

    if (pointCount <= n) {
        result.fill(0);
        const count = new Array(pointCount).fill(0);
        for (let i = 0; i < n; i++) {
            const index = Math.floor(i * (pointCount / n));
            if (downsamplePeaks) {
                result[index] = Math.max(result[index], Math.abs(data[i]));
            } else {
                result[index] += Math.abs(data[i]);
            }
            count[index]++;
        }
        if (!downsamplePeaks) {
            for (let i = 0; i < result.length; i++) {
                result[i] = result[i] / count[i];
            }
        }
    }

    if (memoize) {
        cache[mKey][dKey] = result;
    }
    return result;
}

// Update drawBars function
function drawBars(canvas, ctx, audioData, color) {
    const width = canvas.width;
    const height = canvas.height;
    const marginX = width * 0.1;
    const usableWidth = width - (marginX * 2);

    // Calculate optimal point count and bar dimensions
    const barWidth = 4;
    const barSpacing = 1;
    const pointCount = Math.floor((usableWidth - barSpacing) / (barWidth + barSpacing));

    // Ensure pointCount is even for perfect symmetry
    const adjustedPointCount = pointCount - (pointCount % 2);

    // Normalize the audio data
    const normalizedData = normalizeArray(audioData, adjustedPointCount);

    // Clear background
    ctx.clearRect(0, 0, width, height);

    // Draw center line
    ctx.beginPath();
    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 1;
    ctx.moveTo(marginX, height / 2);
    ctx.lineTo(width - marginX, height / 2);
    ctx.stroke();

    // Calculate center position
    const centerX = width / 2;
    const halfPointCount = adjustedPointCount / 2;

    // Draw bars from center outwards
    for (let i = 0; i < halfPointCount; i++) {
        const amplitude = normalizedData[i];
        const barHeight = Math.max(1, amplitude * height * 0.4);

        // Left side
        const leftX = centerX - (i + 1) * (barWidth + barSpacing);
        const y = (height - barHeight) / 2;

        // Right side
        const rightX = centerX + i * (barWidth + barSpacing);

        // Draw left bar
        ctx.fillStyle = color;
        ctx.fillRect(leftX, y, barWidth, barHeight);

        // Draw right bar
        ctx.fillRect(rightX, y, barWidth, barHeight);

        // Add glow effect for higher amplitudes
        if (amplitude > 0.3) {
            ctx.save();
            ctx.globalAlpha = amplitude - 0.3;
            ctx.shadowColor = color;
            ctx.shadowBlur = 10;
            ctx.fillRect(leftX, y, barWidth, barHeight);
            ctx.fillRect(rightX, y, barWidth, barHeight);
            ctx.restore();
        }
    }
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
            if (audioState.recorder && audioState.recorder.processor && audioState.isRecording) {
                try {
                    const result = audioState.recorder.getFrequencies('voice');
                    // Only draw if we have actual audio data (values above 0)
                    if (result.values.some(value => value > 0)) {
                        drawBars(clientCanvas, clientCtx, result.values, '#0099ff');
                    } else {
                        clientCtx.clearRect(0, 0, clientCanvas.width, clientCanvas.height);
                    }
                } catch (error) {
                    clientCtx.clearRect(0, 0, clientCanvas.width, clientCanvas.height);
                }
            } else {
                clientCtx.clearRect(0, 0, clientCanvas.width, clientCanvas.height);
            }

            // Server-side visualization (playback)
            if (audioState.streamPlayer && audioState.streamPlayer.analyser && audioState.isPlaying) {
                try {
                    const result = audioState.streamPlayer.getFrequencies('voice');
                    // Only draw if we have actual audio data (values above 0)
                    if (result.values.some(value => value > 0)) {
                        drawBars(serverCanvas, serverCtx, result.values, '#ff9900');
                    } else {
                        serverCtx.clearRect(0, 0, serverCanvas.width, serverCanvas.height);
                    }
                } catch (error) {
                    console.error('Visualization error:', error);
                    serverCtx.clearRect(0, 0, serverCanvas.width, serverCanvas.height);
                }
            } else {
                serverCtx.clearRect(0, 0, serverCanvas.width, serverCanvas.height);
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