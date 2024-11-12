// Import all wavtools exports
export { AudioAnalysis, WavPacker, WavStreamPlayer, WavRecorder } from './modules/audio/wavtools/index.js';

// Import and initialize events
import { setupHTMXEvents } from './modules/htmx/htmxEvents.js';

// Initialize HTMX events when the DOM is loaded
document.addEventListener('DOMContentLoaded', function () {
    setupHTMXEvents();
});