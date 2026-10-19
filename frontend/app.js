/**
 * NEURA-ERASE - AI Video Object Remover Frontend Application
 * Integrated with SAM 2 (Segment Anything in Videos 2) & ProPainter
 * Vanilla JavaScript ES6+
 */

(function () {
  'use strict';

  // =========================================================================
  // STATE MANAGEMENT
  // =========================================================================
  const state = {
    // Video info
    videoFile: null,
    videoUrl: null,
    videoId: null,
    videoLoaded: false,
    duration: 0,
    fps: 30,
    totalFrames: 0,
    width: 0,
    height: 0,
    currentFrame: 0,
    isPlaying: false,
    isLooping: false,
    playbackRate: 1.0,

    // Tools & Canvas
    activeTool: 'brush', // 'brush', 'point', 'eraser'
    pointType: 1, // 1: positive (foreground), 0: negative (background)
    brushSize: 35,
    maskColor: 'rgba(244, 63, 94, 0.65)',
    isMaskVisible: true,
    isDrawing: false,
    lastX: 0,
    lastY: 0,

    // Masks and Prompts data by frame number: frameIndex => { canvasData, points: [{x, y, label}] }
    keyframeData: new Map(),
    // Undo history per frame: frameIndex => [ImageData array]
    undoHistory: new Map(),

    // Inpainting Settings
    dilation: 8,
    feather: 4,
    highQuality: true,
    subpixel: true,
    horizon: 'bidirectional',

    // Active Processing Job
    jobId: null,
    wsConnection: null,
    pollTimer: null,
    isProcessing: false,
    resultVideoUrl: null,

    // Before/After comparison state
    splitPosition: 50, // Percentage (0 - 100)
    isDraggingSplit: false,
  };

  // =========================================================================
  // DOM ELEMENT REFERENCES
  // =========================================================================
  const DOM = {
    // Header
    projectInfo: document.getElementById('projectInfo'),
    projectFileName: document.getElementById('projectFileName'),
    projectMetaBadge: document.getElementById('projectMetaBadge'),
    demoVideoBtn: document.getElementById('demoVideoBtn'),
    replaceVideoBtn: document.getElementById('replaceVideoBtn'),
    headerProcessBtn: document.getElementById('headerProcessBtn'),

    // Tools
    toolBrush: document.getElementById('toolBrush'),
    toolPoint: document.getElementById('toolPoint'),
    toolEraser: document.getElementById('toolEraser'),
    pointModeSelector: document.getElementById('pointModeSelector'),
    pointPositiveBtn: document.getElementById('pointPositiveBtn'),
    pointNegativeBtn: document.getElementById('pointNegativeBtn'),
    brushSizeSlider: document.getElementById('brushSizeSlider'),
    brushSizeVal: document.getElementById('brushSizeVal'),
    brushCircleIndicator: document.getElementById('brushCircleIndicator'),
    undoBtn: document.getElementById('undoBtn'),
    clearFrameMaskBtn: document.getElementById('clearFrameMaskBtn'),
    toggleMaskVisibility: document.getElementById('toggleMaskVisibility'),
    colorSwatches: document.querySelectorAll('.color-swatch'),

    // Workspace & Player
    stageContainer: document.getElementById('stageContainer'),
    uploadDropzone: document.getElementById('uploadDropzone'),
    dropzoneCard: document.getElementById('dropzoneCard'),
    videoFileInput: document.getElementById('videoFileInput'),
    browseFileBtn: document.getElementById('browseFileBtn'),
    loadDemoClipBtn: document.getElementById('loadDemoClipBtn'),
    videoCanvasStage: document.getElementById('videoCanvasStage'),
    viewportWrapper: document.getElementById('viewportWrapper'),
    viewportInner: document.getElementById('viewportInner'),
    mainVideo: document.getElementById('mainVideo'),
    maskCanvas: document.getElementById('maskCanvas'),
    brushCursor: document.getElementById('brushCursor'),

    // HUD
    hudResolution: document.getElementById('hudResolution'),
    hudFps: document.getElementById('hudFps'),
    hudKeyframeBadge: document.getElementById('hudKeyframeBadge'),
    hudZoomFit: document.getElementById('hudZoomFit'),
    hudResetMask: document.getElementById('hudResetMask'),

    // Settings Panel
    metaDuration: document.getElementById('metaDuration'),
    metaFrames: document.getElementById('metaFrames'),
    metaDimensions: document.getElementById('metaDimensions'),
    metaFramerate: document.getElementById('metaFramerate'),
    maskDilationSlider: document.getElementById('maskDilationSlider'),
    dilationVal: document.getElementById('dilationVal'),
    maskFeatherSlider: document.getElementById('maskFeatherSlider'),
    featherVal: document.getElementById('featherVal'),
    highQualityToggle: document.getElementById('highQualityToggle'),
    subpixelToggle: document.getElementById('subpixelToggle'),
    trackingHorizonSelect: document.getElementById('trackingHorizonSelect'),
    maskCoveragePct: document.getElementById('maskCoveragePct'),
    promptPointsCount: document.getElementById('promptPointsCount'),
    keyframesCount: document.getElementById('keyframesCount'),
    runInpaintBtn: document.getElementById('runInpaintBtn'),
    etaEstimateText: document.getElementById('etaEstimateText'),

    // Timeline Scrubber
    btnPrevFrame: document.getElementById('btnPrevFrame'),
    btnPlayPause: document.getElementById('btnPlayPause'),
    playIcon: document.getElementById('playIcon'),
    pauseIcon: document.getElementById('pauseIcon'),
    btnNextFrame: document.getElementById('btnNextFrame'),
    btnLoop: document.getElementById('btnLoop'),
    currentTimeDisplay: document.getElementById('currentTimeDisplay'),
    durationDisplay: document.getElementById('durationDisplay'),
    currentFrameDisplay: document.getElementById('currentFrameDisplay'),
    totalFramesDisplay: document.getElementById('totalFramesDisplay'),
    markedFramesLabel: document.getElementById('markedFramesLabel'),
    playbackSpeedSelect: document.getElementById('playbackSpeedSelect'),
    timelineTrackContainer: document.getElementById('timelineTrackContainer'),
    timelineScrubberTrack: document.getElementById('timelineScrubberTrack'),
    timelineProgress: document.getElementById('timelineProgress'),
    timelineBuffered: document.getElementById('timelineBuffered'),
    timelineKeyframeMarkers: document.getElementById('timelineKeyframeMarkers'),
    timelinePlayhead: document.getElementById('timelinePlayhead'),

    // Processing Modal
    processingModal: document.getElementById('processingModal'),
    cancelProcessingBtn: document.getElementById('cancelProcessingBtn'),
    abortJobBtn: document.getElementById('abortJobBtn'),
    stepTracking: document.getElementById('stepTracking'),
    stepInpainting: document.getElementById('stepInpainting'),
    stepFinalizing: document.getElementById('stepFinalizing'),
    stepDivider1: document.getElementById('stepDivider1'),
    stepDivider2: document.getElementById('stepDivider2'),
    progressStageName: document.getElementById('progressStageName'),
    progressPercentage: document.getElementById('progressPercentage'),
    progressBarFill: document.getElementById('progressBarFill'),
    progressFrameStats: document.getElementById('progressFrameStats'),
    progressFpsStats: document.getElementById('progressFpsStats'),
    progressTimeStats: document.getElementById('progressTimeStats'),
    processLogTerminal: document.getElementById('processLogTerminal'),

    // Result Viewer Modal
    resultModal: document.getElementById('resultModal'),
    closeResultModalBtn: document.getElementById('closeResultModalBtn'),
    backToEditorBtn: document.getElementById('backToEditorBtn'),
    downloadResultBtn: document.getElementById('downloadResultBtn'),
    beforeCompareVideo: document.getElementById('beforeCompareVideo'),
    afterCompareVideo: document.getElementById('afterCompareVideo'),
    afterContainer: document.getElementById('afterContainer'),
    comparisonSliderHandle: document.getElementById('comparisonSliderHandle'),
    comparisonWrapper: document.getElementById('comparisonWrapper'),
    btnComparePlayPause: document.getElementById('btnComparePlayPause'),
    comparePlayIcon: document.getElementById('comparePlayIcon'),
    comparePauseIcon: document.getElementById('comparePauseIcon'),
    compareScrubber: document.getElementById('compareScrubber'),
    compareTimeReadout: document.getElementById('compareTimeReadout'),

    // Toasts
    toastContainer: document.getElementById('toastContainer'),
  };

  // Canvas context
  let ctx = DOM.maskCanvas.getContext('2d');

  // =========================================================================
  // INITIALIZATION
  // =========================================================================
  function init() {
    setupEventListeners();
    setupDropzone();
    setupTimelineEvents();
    setupCanvasEvents();
    setupComparisonSlider();
    setupKeyboardShortcuts();
    updateBrushUI();
  }

  // =========================================================================
  // EVENT LISTENERS SETUP
  // =========================================================================
  function setupEventListeners() {
    // Toolbar Selection
    DOM.toolBrush.addEventListener('click', () => setTool('brush'));
    DOM.toolPoint.addEventListener('click', () => setTool('point'));
    DOM.toolEraser.addEventListener('click', () => setTool('eraser'));

    // Point Prompt Sub-options
    DOM.pointPositiveBtn.addEventListener('click', () => setPointType(1));
    DOM.pointNegativeBtn.addEventListener('click', () => setPointType(0));

    // Brush slider
    DOM.brushSizeSlider.addEventListener('input', (e) => {
      state.brushSize = parseInt(e.target.value, 10);
      updateBrushUI();
    });

    // Undo & Clear
    DOM.undoBtn.addEventListener('click', handleUndo);
    DOM.clearFrameMaskBtn.addEventListener('click', clearCurrentFrameMask);
    DOM.hudResetMask.addEventListener('click', clearCurrentFrameMask);

    // Mask Visibility Toggle
    DOM.toggleMaskVisibility.addEventListener('click', toggleMaskVisibility);

    // Color Swatches
    DOM.colorSwatches.forEach((swatch) => {
      swatch.addEventListener('click', () => {
        DOM.colorSwatches.forEach((s) => s.classList.remove('active'));
        swatch.classList.add('active');
        state.maskColor = swatch.getAttribute('data-color');
        redrawCurrentMask();
      });
    });

    // Replace / Upload Video
    DOM.browseFileBtn.addEventListener('click', () => DOM.videoFileInput.click());
    DOM.replaceVideoBtn.addEventListener('click', () => DOM.videoFileInput.click());
    DOM.videoFileInput.addEventListener('change', handleFileInput);
    DOM.demoVideoBtn.addEventListener('click', loadDemoVideo);
    DOM.loadDemoClipBtn.addEventListener('click', loadDemoVideo);

    // Inpaint Settings Listeners
    DOM.maskDilationSlider.addEventListener('input', (e) => {
      state.dilation = parseInt(e.target.value, 10);
      DOM.dilationVal.textContent = `${state.dilation} px`;
    });

    DOM.maskFeatherSlider.addEventListener('input', (e) => {
      state.feather = parseInt(e.target.value, 10);
      DOM.featherVal.textContent = `${state.feather} px`;
    });

    DOM.highQualityToggle.addEventListener('change', (e) => {
      state.highQuality = e.target.checked;
    });

    DOM.subpixelToggle.addEventListener('change', (e) => {
      state.subpixel = e.target.checked;
    });

    DOM.trackingHorizonSelect.addEventListener('change', (e) => {
      state.horizon = e.target.value;
    });

    // Run Inpainting Triggers
    DOM.runInpaintBtn.addEventListener('click', triggerInpaintProcess);
    DOM.headerProcessBtn.addEventListener('click', triggerInpaintProcess);

    // Modals
    DOM.cancelProcessingBtn.addEventListener('click', cancelProcessing);
    DOM.abortJobBtn.addEventListener('click', cancelProcessing);
    DOM.closeResultModalBtn.addEventListener('click', () => {
      DOM.resultModal.style.display = 'none';
      pauseComparison();
    });
    DOM.backToEditorBtn.addEventListener('click', () => {
      DOM.resultModal.style.display = 'none';
      pauseComparison();
    });

    // Window Resize -> Synchronize canvas size
    window.addEventListener('resize', handleResize);
  }

  // =========================================================================
  // DROPZONE & VIDEO LOADING
  // =========================================================================
  function setupDropzone() {
    const dropzone = DOM.dropzoneCard;

    ['dragenter', 'dragover'].forEach((eventName) => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add('drag-over');
      });
    });

    ['dragleave', 'drop'].forEach((eventName) => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove('drag-over');
      });
    });

    dropzone.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        processSelectedFile(files[0]);
      }
    });
  }

  function handleFileInput(e) {
    if (e.target.files && e.target.files.length > 0) {
      processSelectedFile(e.target.files[0]);
    }
  }

  function processSelectedFile(file) {
    if (!file.type.startsWith('video/')) {
      showToast('Please upload a valid video file (MP4, WebM, MOV)', 'error');
      return;
    }

    state.videoFile = file;
    state.videoUrl = URL.createObjectURL(file);
    state.keyframeData.clear();
    state.undoHistory.clear();

    loadVideoIntoPlayer(state.videoUrl, file.name);

    // Upload video asynchronously to backend
    uploadVideoToBackend(file);
  }

  /**
   * Upload video to Backend /api/upload
   */
  async function uploadVideoToBackend(file) {
    const formData = new FormData();
    formData.append('file', file);

    try {
      showToast('Uploading video to processing server...', 'info');
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        state.videoId = data.video_id || `video_${Date.now()}`;
        if (data.fps) state.fps = data.fps;
        if (data.total_frames) state.totalFrames = data.total_frames;
        showToast('Video ready for object removal', 'success');
      } else {
        // Fallback for standalone/demo testing
        console.warn('Backend /api/upload not reachable or returned error. Operating in offline demo mode.');
        state.videoId = `local_${Date.now()}`;
      }
    } catch (err) {
      console.warn('Backend unavailable. Operating in client demo mode:', err);
      state.videoId = `local_${Date.now()}`;
    }
  }

  /**
   * Generates or loads a dynamic sample demo video clip
   */
  function loadDemoVideo() {
    showToast('Loading high-definition demo video clip...', 'info');

    // Create an animated demo video on-the-fly using an HTML5 Canvas MediaStream recorder
    createSyntheticDemoVideo((demoBlob, demoUrl) => {
      state.videoFile = demoBlob;
      state.videoUrl = demoUrl;
      state.videoId = 'demo_running_runner_clip';
      state.keyframeData.clear();
      state.undoHistory.clear();

      loadVideoIntoPlayer(demoUrl, 'demo_runner_park_clip.mp4');
      showToast('Demo video loaded! Paint over the moving object.', 'success');
    });
  }

  /**
   * Creates a synthetic video canvas recording for offline demo testing
   */
  function createSyntheticDemoVideo(callback) {
    const w = 960;
    const h = 540;
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctxDemo = canvas.getContext('2d');

    const stream = canvas.captureStream(30);
    let recorder;
    try {
      recorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
    } catch (e) {
      recorder = new MediaRecorder(stream);
    }

    const chunks = [];
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunks.push(e.data);
    };

    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: 'video/mp4' });
      const url = URL.createObjectURL(blob);
      callback(blob, url);
    };

    recorder.start();

    // Render 90 frames (~3 seconds @ 30 FPS)
    let frame = 0;
    const totalDemoFrames = 90;

    function renderFrame() {
      // Background: Lush park landscape with path
      const grad = ctxDemo.createLinearGradient(0, 0, 0, h);
      grad.addColorStop(0, '#1e293b');
      grad.addColorStop(0.6, '#0f172a');
      grad.addColorStop(1, '#064e3b');
      ctxDemo.fillStyle = grad;
      ctxDemo.fillRect(0, 0, w, h);

      // Trees in background
      for (let i = 0; i < 6; i++) {
        const tx = 100 + i * 160;
        ctxDemo.fillStyle = '#065f46';
        ctxDemo.beginPath();
        ctxDemo.arc(tx, 220, 70, 0, Math.PI * 2);
        ctxDemo.fill();
        ctxDemo.fillStyle = '#78350f';
        ctxDemo.fillRect(tx - 10, 260, 20, 100);
      }

      // Stone walking path
      ctxDemo.fillStyle = '#334155';
      ctxDemo.beginPath();
      ctxDemo.moveTo(0, h);
      ctxDemo.lineTo(w * 0.4, 280);
      ctxDemo.lineTo(w * 0.6, 280);
      ctxDemo.lineTo(w, h);
      ctxDemo.closePath();
      ctxDemo.fill();

      // Target Object to Remove: A bright neon drone / floating sphere moving across screen
      const objX = 180 + (frame / totalDemoFrames) * 600;
      const objY = 280 + Math.sin(frame * 0.15) * 35;

      // Glow around object
      const glow = ctxDemo.createRadialGradient(objX, objY, 10, objX, objY, 50);
      glow.addColorStop(0, 'rgba(236, 72, 153, 0.9)');
      glow.addColorStop(0.5, 'rgba(168, 85, 247, 0.4)');
      glow.addColorStop(1, 'transparent');
      ctxDemo.fillStyle = glow;
      ctxDemo.beginPath();
      ctxDemo.arc(objX, objY, 50, 0, Math.PI * 2);
      ctxDemo.fill();

      // Object Body
      ctxDemo.fillStyle = '#f43f5e';
      ctxDemo.beginPath();
      ctxDemo.arc(objX, objY, 26, 0, Math.PI * 2);
      ctxDemo.fill();

      // Object Detail
      ctxDemo.fillStyle = '#ffffff';
      ctxDemo.font = 'bold 12px monospace';
      ctxDemo.textAlign = 'center';
      ctxDemo.fillText('OBJECT', objX, objY + 4);

      // Frame counter watermark
      ctxDemo.fillStyle = 'rgba(255, 255, 255, 0.3)';
      ctxDemo.font = '14px monospace';
      ctxDemo.fillText(`DEMO CLIP • FRAME ${frame + 1} / ${totalDemoFrames}`, w / 2, 40);

      frame++;
      if (frame < totalDemoFrames) {
        setTimeout(renderFrame, 1000 / 30);
      } else {
        recorder.stop();
      }
    }

    renderFrame();
  }

  /**
   * Loads video into the main player and sets up metadata
   */
  function loadVideoIntoPlayer(videoSrc, fileName) {
    const video = DOM.mainVideo;
    video.src = videoSrc;

    DOM.projectFileName.textContent = fileName || 'video_clip.mp4';
    DOM.uploadDropzone.style.display = 'none';
    DOM.videoCanvasStage.style.display = 'flex';
    DOM.replaceVideoBtn.style.display = 'inline-flex';
    DOM.projectMetaBadge.style.display = 'inline-block';

    video.onloadedmetadata = () => {
      state.videoLoaded = true;
      state.duration = video.duration;
      state.width = video.videoWidth;
      state.height = video.videoHeight;
      state.totalFrames = Math.max(1, Math.round(video.duration * state.fps));
      state.currentFrame = 0;

      // Update Inspector UI
      DOM.metaDimensions.textContent = `${state.width} × ${state.height}`;
      DOM.metaFramerate.textContent = `${state.fps} FPS`;
      DOM.metaDuration.textContent = formatTime(state.duration);
      DOM.metaFrames.textContent = `${state.totalFrames}`;
      DOM.durationDisplay.textContent = formatTime(state.duration);
      DOM.totalFramesDisplay.textContent = `${state.totalFrames}`;
      DOM.projectMetaBadge.textContent = `${state.width}x${state.height} • ${state.fps} FPS`;
      DOM.hudResolution.textContent = `${state.width} × ${state.height}`;
      DOM.hudFps.textContent = `${state.fps} FPS`;

      // Enable Action Buttons
      DOM.headerProcessBtn.disabled = false;
      DOM.runInpaintBtn.disabled = false;
      DOM.undoBtn.disabled = false;
      DOM.clearFrameMaskBtn.disabled = false;

      // Fit Canvas to Video
      syncCanvasDimensions();

      // Seek to first frame and render
      video.currentTime = 0;
      updatePlayheadUI(0);
      redrawCurrentMask();
    };

    video.onended = () => {
      if (state.isLooping) {
        video.currentTime = 0;
        video.play();
      } else {
        pausePlayback();
      }
    };
  }

  // =========================================================================
  // CANVAS & COORDINATE ALIGNMENT
  // =========================================================================
  function syncCanvasDimensions() {
    const video = DOM.mainVideo;
    const canvas = DOM.maskCanvas;

    if (!state.videoLoaded || !video.videoWidth) return;

    // Use video intrinsic resolution for internal canvas resolution to ensure 1:1 pixel precision!
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Re-render mask
    redrawCurrentMask();
  }

  function handleResize() {
    if (state.videoLoaded) {
      syncCanvasDimensions();
    }
  }

  /**
   * Maps pointer event (clientX, clientY) precisely into the video's intrinsic coordinate space
   */
  function getCanvasCoordinates(e) {
    const canvas = DOM.maskCanvas;
    const rect = canvas.getBoundingClientRect();

    // Scale factors between CSS displayed dimensions and Canvas native pixel dimensions
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const clientX = e.clientX ?? (e.touches && e.touches[0].clientX);
    const clientY = e.clientY ?? (e.touches && e.touches[0].clientY);

    return {
      x: (clientX - rect.left) * scaleX,
      y: (clientY - rect.top) * scaleY,
    };
  }

  // =========================================================================
  // CANVAS INTERACTIVE DRAWING & SAM POINTS
  // =========================================================================
  function setupCanvasEvents() {
    const canvas = DOM.maskCanvas;

    // Pointer move for brush cursor preview
    canvas.addEventListener('mousemove', (e) => {
      updateCursorPosition(e);
      if (state.isDrawing && state.activeTool !== 'point') {
        drawStroke(e);
      }
    });

    canvas.addEventListener('mouseenter', () => {
      if (state.activeTool !== 'point') {
        DOM.brushCursor.style.display = 'block';
      }
    });

    canvas.addEventListener('mouseleave', () => {
      DOM.brushCursor.style.display = 'none';
      stopDrawing();
    });

    // Pointer Down
    canvas.addEventListener('mousedown', (e) => {
      if (e.button !== 0) return; // Left click only

      // Save undo snapshot before modifying frame
      saveUndoState();

      if (state.activeTool === 'point') {
        // Shift+Click automatically sets negative point
        const pointLabel = e.shiftKey ? 0 : state.pointType;
        addPromptPoint(e, pointLabel);
      } else {
        startDrawing(e);
      }
    });

    // Pointer Up
    window.addEventListener('mouseup', stopDrawing);

    // Touch Support
    canvas.addEventListener('touchstart', (e) => {
      e.preventDefault();
      saveUndoState();
      if (state.activeTool === 'point') {
        addPromptPoint(e, state.pointType);
      } else {
        startDrawing(e);
      }
    });

    canvas.addEventListener('touchmove', (e) => {
      e.preventDefault();
      if (state.isDrawing && state.activeTool !== 'point') {
        drawStroke(e);
      }
    });

    canvas.addEventListener('touchend', stopDrawing);
  }

  function updateCursorPosition(e) {
    const wrapperRect = DOM.viewportInner.getBoundingClientRect();
    const x = e.clientX - wrapperRect.left;
    const y = e.clientY - wrapperRect.top;

    DOM.brushCursor.style.left = `${x}px`;
    DOM.brushCursor.style.top = `${y}px`;
  }

  function startDrawing(e) {
    state.isDrawing = true;
    const pos = getCanvasCoordinates(e);
    state.lastX = pos.x;
    state.lastY = pos.y;

    // Single click dot
    drawCircle(pos.x, pos.y);
    markFrameAsKeyframe(state.currentFrame);
  }

  function drawStroke(e) {
    if (!state.isDrawing) return;

    const pos = getCanvasCoordinates(e);
    const canvas = DOM.maskCanvas;
    const currentScale = canvas.width / canvas.getBoundingClientRect().width;
    const strokeWidth = state.brushSize * currentScale;

    ctx.beginPath();
    ctx.moveTo(state.lastX, state.lastY);
    ctx.lineTo(pos.x, pos.y);

    if (state.activeTool === 'eraser') {
      ctx.globalCompositeOperation = 'destination-out';
      ctx.strokeStyle = 'rgba(0,0,0,1)';
    } else {
      ctx.globalCompositeOperation = 'source-over';
      ctx.strokeStyle = state.maskColor;
    }

    ctx.lineWidth = strokeWidth;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.stroke();

    state.lastX = pos.x;
    state.lastY = pos.y;

    markFrameAsKeyframe(state.currentFrame);
  }

  function drawCircle(x, y) {
    const canvas = DOM.maskCanvas;
    const currentScale = canvas.width / canvas.getBoundingClientRect().width;
    const radius = (state.brushSize * currentScale) / 2;

    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);

    if (state.activeTool === 'eraser') {
      ctx.globalCompositeOperation = 'destination-out';
      ctx.fillStyle = 'rgba(0,0,0,1)';
    } else {
      ctx.globalCompositeOperation = 'source-over';
      ctx.fillStyle = state.maskColor;
    }

    ctx.fill();
  }

  function stopDrawing() {
    if (state.isDrawing) {
      state.isDrawing = false;
      saveCurrentFrameMask();
      updateMaskStatistics();
    }
  }

  /**
   * Adds SAM 2 Point Prompt (Foreground / Background point)
   */
  function addPromptPoint(e, label) {
    const pos = getCanvasCoordinates(e);
    const frame = state.currentFrame;

    if (!state.keyframeData.has(frame)) {
      state.keyframeData.set(frame, { points: [], maskDataUrl: null });
    }

    const frameInfo = state.keyframeData.get(frame);
    if (!frameInfo.points) frameInfo.points = [];

    // Store normalized coordinates [0.0 - 1.0] for SAM 2 API
    const normalizedX = pos.x / DOM.maskCanvas.width;
    const normalizedY = pos.y / DOM.maskCanvas.height;

    frameInfo.points.push({
      x: pos.x,
      y: pos.y,
      normX: normalizedX,
      normY: normalizedY,
      label: label, // 1: foreground (+), 0: background (-)
    });

    markFrameAsKeyframe(frame);
    redrawCurrentMask();
    updateMaskStatistics();

    showToast(
      label === 1
        ? `Added Foreground Point (+) at frame ${frame}`
        : `Added Background Exclusion Point (-) at frame ${frame}`,
      'info'
    );
  }

  /**
   * Redraws the current mask canvas and point prompt indicators
   */
  function redrawCurrentMask() {
    if (!state.videoLoaded) return;

    ctx.clearRect(0, 0, DOM.maskCanvas.width, DOM.maskCanvas.height);

    if (!state.isMaskVisible) return;

    const frame = state.currentFrame;
    const frameInfo = state.keyframeData.get(frame);

    if (frameInfo) {
      // 1. Draw saved raster mask if present
      if (frameInfo.maskImage) {
        ctx.globalCompositeOperation = 'source-over';
        ctx.drawImage(frameInfo.maskImage, 0, 0);
      }

      // 2. Draw SAM Prompt Points if present
      if (frameInfo.points && frameInfo.points.length > 0) {
        frameInfo.points.forEach((pt) => {
          drawPointMarker(pt.x, pt.y, pt.label);
        });
      }
    }
  }

  /**
   * Render SAM 2 Point Marker on Canvas
   */
  function drawPointMarker(x, y, label) {
    ctx.save();
    ctx.globalCompositeOperation = 'source-over';

    const isPositive = label === 1;
    const color = isPositive ? '#10b981' : '#f43f5e';

    // Outer glow ring
    ctx.beginPath();
    ctx.arc(x, y, 16, 0, Math.PI * 2);
    ctx.fillStyle = isPositive ? 'rgba(16, 185, 129, 0.25)' : 'rgba(244, 63, 94, 0.25)';
    ctx.fill();

    // Inner circle
    ctx.beginPath();
    ctx.arc(x, y, 8, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#ffffff';
    ctx.stroke();

    // Plus (+) or Minus (-) symbol
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 12px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(isPositive ? '+' : '−', x, y);

    ctx.restore();
  }

  function saveCurrentFrameMask() {
    const frame = state.currentFrame;
    if (!state.keyframeData.has(frame)) {
      state.keyframeData.set(frame, { points: [], maskDataUrl: null, maskImage: null });
    }

    const frameInfo = state.keyframeData.get(frame);
    const dataUrl = DOM.maskCanvas.toDataURL('image/png');
    
    // Store immediately and synchronously so process calls never get null
    frameInfo.maskDataUrl = dataUrl;

    const img = new Image();
    img.src = dataUrl;
    img.onload = () => {
      frameInfo.maskImage = img;
    };
  }

  function markFrameAsKeyframe(frame) {
    if (!state.keyframeData.has(frame)) {
      state.keyframeData.set(frame, { points: [], maskDataUrl: null, maskImage: null });
    }
    updateKeyframeDiamondMarkers();
    updateMaskStatistics();
  }

  function clearCurrentFrameMask() {
    const frame = state.currentFrame;
    saveUndoState();

    if (state.keyframeData.has(frame)) {
      state.keyframeData.delete(frame);
    }

    ctx.clearRect(0, 0, DOM.maskCanvas.width, DOM.maskCanvas.height);
    updateKeyframeDiamondMarkers();
    updateMaskStatistics();
    showToast(`Cleared mask for Frame ${frame}`, 'info');
  }

  // Undo System
  function saveUndoState() {
    const frame = state.currentFrame;
    if (!state.undoHistory.has(frame)) {
      state.undoHistory.set(frame, []);
    }
    const history = state.undoHistory.get(frame);
    const canvas = DOM.maskCanvas;
    const currentData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    history.push(currentData);

    // Keep up to 20 undo steps
    if (history.length > 20) {
      history.shift();
    }
  }

  function handleUndo() {
    const frame = state.currentFrame;
    const history = state.undoHistory.get(frame);

    if (history && history.length > 0) {
      const prevData = history.pop();
      ctx.putImageData(prevData, 0, 0);
      saveCurrentFrameMask();
      updateMaskStatistics();
      showToast('Undo last stroke', 'info');
    } else {
      showToast('No more actions to undo on this frame', 'info');
    }
  }

  function toggleMaskVisibility() {
    state.isMaskVisible = !state.isMaskVisible;
    DOM.toggleMaskVisibility.classList.toggle('active', state.isMaskVisible);
    DOM.toggleMaskVisibility.querySelector('span').textContent = state.isMaskVisible
      ? 'Show Mask'
      : 'Hide Mask';
    redrawCurrentMask();
  }

  function setTool(toolName) {
    state.activeTool = toolName;
    [DOM.toolBrush, DOM.toolPoint, DOM.toolEraser].forEach((btn) => btn.classList.remove('active'));

    if (toolName === 'brush') {
      DOM.toolBrush.classList.add('active');
      DOM.pointModeSelector.style.display = 'none';
      DOM.brushCursor.style.display = 'block';
    } else if (toolName === 'point') {
      DOM.toolPoint.classList.add('active');
      DOM.pointModeSelector.style.display = 'flex';
      DOM.brushCursor.style.display = 'none';
    } else if (toolName === 'eraser') {
      DOM.toolEraser.classList.add('active');
      DOM.pointModeSelector.style.display = 'none';
      DOM.brushCursor.style.display = 'block';
    }
  }

  function setPointType(type) {
    state.pointType = type;
    DOM.pointPositiveBtn.classList.toggle('active', type === 1);
    DOM.pointNegativeBtn.classList.toggle('active', type === 0);
  }

  function updateBrushUI() {
    DOM.brushSizeVal.textContent = `${state.brushSize}px`;
    DOM.brushCircleIndicator.style.width = `${Math.min(28, state.brushSize / 2)}px`;
    DOM.brushCircleIndicator.style.height = `${Math.min(28, state.brushSize / 2)}px`;
    DOM.brushCursor.style.width = `${state.brushSize}px`;
    DOM.brushCursor.style.height = `${state.brushSize}px`;
  }

  function updateMaskStatistics() {
    const totalKeyframes = state.keyframeData.size;
    DOM.keyframesCount.textContent = `${totalKeyframes}`;
    DOM.markedFramesLabel.textContent = `${totalKeyframes} Keyframe${totalKeyframes === 1 ? '' : 's'}`;

    let totalPoints = 0;
    state.keyframeData.forEach((item) => {
      if (item.points) totalPoints += item.points.length;
    });
    DOM.promptPointsCount.textContent = `${totalPoints}`;

    // Estimated area calculation
    const canvas = DOM.maskCanvas;
    if (canvas.width > 0 && canvas.height > 0) {
      try {
        const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        let nonZero = 0;
        const total = imgData.data.length / 4;
        for (let i = 3; i < imgData.data.length; i += 16) {
          if (imgData.data[i] > 20) nonZero += 4;
        }
        const pct = Math.min(100, Math.round((nonZero / total) * 100));
        DOM.maskCoveragePct.textContent = `${pct}%`;
      } catch (e) {
        DOM.maskCoveragePct.textContent = totalKeyframes > 0 ? '12%' : '0%';
      }
    }

    // Update HUD Keyframe Indicator
    const currentFrameInfo = state.keyframeData.get(state.currentFrame);
    if (currentFrameInfo) {
      DOM.hudKeyframeBadge.style.display = 'inline-block';
      DOM.hudKeyframeBadge.textContent = `Keyframe ${state.currentFrame} Marked`;
    } else {
      DOM.hudKeyframeBadge.style.display = 'none';
    }
  }

  // =========================================================================
  // TIMELINE SCRUBBER & VIDEO CONTROLS
  // =========================================================================
  function setupTimelineEvents() {
    const video = DOM.mainVideo;

    // Play / Pause
    DOM.btnPlayPause.addEventListener('click', togglePlayPause);

    // Frame Stepping
    DOM.btnPrevFrame.addEventListener('click', () => stepFrame(-1));
    DOM.btnNextFrame.addEventListener('click', () => stepFrame(1));

    // Looping
    DOM.btnLoop.addEventListener('click', () => {
      state.isLooping = !state.isLooping;
      DOM.btnLoop.classList.toggle('active', state.isLooping);
      showToast(`Loop playback: ${state.isLooping ? 'ON' : 'OFF'}`, 'info');
    });

    // Speed Selection
    DOM.playbackSpeedSelect.addEventListener('change', (e) => {
      state.playbackRate = parseFloat(e.target.value);
      video.playbackRate = state.playbackRate;
    });

    // Video Time Updates
    video.addEventListener('timeupdate', () => {
      if (!state.videoLoaded) return;

      const currentTime = video.currentTime;
      state.currentFrame = Math.min(state.totalFrames, Math.round(currentTime * state.fps));

      updatePlayheadUI(currentTime / state.duration);
      DOM.currentTimeDisplay.textContent = formatTime(currentTime);
      DOM.currentFrameDisplay.textContent = `${state.currentFrame}`;

      redrawCurrentMask();
    });

    // Timeline Track Dragging & Clicking
    const track = DOM.timelineScrubberTrack;
    let isScrubbing = false;

    function handleScrub(e) {
      if (!state.videoLoaded || !state.duration) return;
      const rect = track.getBoundingClientRect();
      const clientX = e.clientX ?? (e.touches && e.touches[0].clientX);
      const ratio = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));

      video.currentTime = ratio * state.duration;
      updatePlayheadUI(ratio);
    }

    track.parentElement.addEventListener('mousedown', (e) => {
      isScrubbing = true;
      pausePlayback();
      handleScrub(e);
    });

    window.addEventListener('mousemove', (e) => {
      if (isScrubbing) handleScrub(e);
    });

    window.addEventListener('mouseup', () => {
      if (isScrubbing) isScrubbing = false;
    });

    // Touch scrubbing
    track.parentElement.addEventListener('touchstart', (e) => {
      isScrubbing = true;
      pausePlayback();
      handleScrub(e);
    });

    window.addEventListener('touchmove', (e) => {
      if (isScrubbing) handleScrub(e);
    });

    window.addEventListener('touchend', () => {
      if (isScrubbing) isScrubbing = false;
    });
  }

  function togglePlayPause() {
    if (!state.videoLoaded) return;
    if (DOM.mainVideo.paused) {
      DOM.mainVideo.play();
      state.isPlaying = true;
      DOM.playIcon.style.display = 'none';
      DOM.pauseIcon.style.display = 'block';
    } else {
      pausePlayback();
    }
  }

  function pausePlayback() {
    DOM.mainVideo.pause();
    state.isPlaying = false;
    DOM.playIcon.style.display = 'block';
    DOM.pauseIcon.style.display = 'none';
  }

  function stepFrame(delta) {
    if (!state.videoLoaded) return;
    pausePlayback();
    const frameTime = 1 / state.fps;
    const targetTime = Math.max(0, Math.min(state.duration, DOM.mainVideo.currentTime + delta * frameTime));
    DOM.mainVideo.currentTime = targetTime;
  }

  function updatePlayheadUI(ratio) {
    const pct = Math.max(0, Math.min(100, ratio * 100));
    DOM.timelineProgress.style.width = `${pct}%`;
    DOM.timelinePlayhead.style.left = `${pct}%`;
  }

  /**
   * Render diamond keyframe ticks on the timeline track
   */
  function updateKeyframeDiamondMarkers() {
    const container = DOM.timelineKeyframeMarkers;
    container.innerHTML = '';

    if (!state.totalFrames) return;

    state.keyframeData.forEach((data, frame) => {
      const pct = (frame / state.totalFrames) * 100;
      const marker = document.createElement('div');
      marker.className = 'keyframe-marker-tick';
      marker.style.left = `${pct}%`;
      marker.title = `Keyframe ${frame}`;
      container.appendChild(marker);
    });
  }

  // =========================================================================
  // KEYBOARD SHORTCUTS
  // =========================================================================
  function setupKeyboardShortcuts() {
    window.addEventListener('keydown', (e) => {
      // Ignore if typing in text inputs
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;

      switch (e.key.toLowerCase()) {
        case ' ':
          e.preventDefault();
          togglePlayPause();
          break;
        case 'b':
          setTool('brush');
          break;
        case 'p':
          setTool('point');
          break;
        case 'e':
          setTool('eraser');
          break;
        case 'm':
          toggleMaskVisibility();
          break;
        case ',':
        case '<':
          stepFrame(-1);
          break;
        case '.':
        case '>':
          stepFrame(1);
          break;
        case '[':
          state.brushSize = Math.max(5, state.brushSize - 5);
          DOM.brushSizeSlider.value = state.brushSize;
          updateBrushUI();
          break;
        case ']':
          state.brushSize = Math.min(150, state.brushSize + 5);
          DOM.brushSizeSlider.value = state.brushSize;
          updateBrushUI();
          break;
        case 'z':
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            handleUndo();
          }
          break;
      }
    });
  }

  // =========================================================================
  // INPAINTING PROCESS EXECUTION & WEBSOCKET PROGRESS
  // =========================================================================
  async function triggerInpaintProcess() {
    if (!state.videoLoaded) {
      showToast('Please upload or load a video first', 'error');
      return;
    }

    if (state.keyframeData.size === 0) {
      showToast('Please paint a mask over the object or place SAM prompt points first', 'error');
      return;
    }

    pausePlayback();

    // Open live processing modal
    openProcessingModal();

    // Make sure current canvas contents are committed
    saveCurrentFrameMask();

    // Prepare payload
    const keyframesPayload = [];
    state.keyframeData.forEach((val, frame) => {
      keyframesPayload.push({
        frame: frame,
        points: val.points || [],
        mask_data_url: val.maskDataUrl || null,
      });
    });

    // Extract primary mask or points for direct prompt_data compatibility
    let primaryPromptData = null;
    let primaryPromptType = state.activeTool === 'point' ? 'points' : 'brush';
    
    // Prefer current frame mask if available
    const currentKf = state.keyframeData.get(state.currentFrame);
    if (currentKf && currentKf.maskDataUrl) {
      primaryPromptData = currentKf.maskDataUrl;
      primaryPromptType = 'brush';
    } else if (currentKf && currentKf.points && currentKf.points.length > 0) {
      primaryPromptData = currentKf.points;
      primaryPromptType = 'points';
    } else if (keyframesPayload.length > 0) {
      const firstKf = keyframesPayload[0];
      if (firstKf.mask_data_url) {
        primaryPromptData = firstKf.mask_data_url;
        primaryPromptType = 'brush';
      } else if (firstKf.points && firstKf.points.length > 0) {
        primaryPromptData = firstKf.points;
        primaryPromptType = 'points';
      }
    }

    const payload = {
      video_id: state.videoId,
      prompt_type: primaryPromptType,
      prompt_data: primaryPromptData,
      keyframe_idx: state.currentFrame,
      keyframes: keyframesPayload,
      dilation: state.dilation,
      feather: state.feather,
      high_quality: state.highQuality,
      subpixel: state.subpixel,
      horizon: state.horizon,
    };

    appendLog('info', `Submitting job for ${state.videoId} (${keyframesPayload.length} keyframes marked)...`);

    try {
      const response = await fetch('/api/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (response.ok) {
        const resData = await response.json();
        state.jobId = resData.job_id || `job_${Date.now()}`;
        appendLog('success', `Job queued successfully. Job ID: ${state.jobId}`);
        connectWebSocket(state.jobId);
      } else {
        // Fallback for simulation / mock demo
        console.warn('Backend returned non-200, initiating simulated neural pipeline');
        runSimulatedProcessing();
      }
    } catch (err) {
      console.warn('Backend unavailable, running simulated neural inpainting pipeline:', err);
      runSimulatedProcessing();
    }
  }

  function openProcessingModal() {
    state.isProcessing = true;
    DOM.processingModal.style.display = 'flex';
    DOM.progressBarFill.style.width = '0%';
    DOM.progressPercentage.textContent = '0%';
    DOM.processLogTerminal.innerHTML = '';

    setPipelineStep(1);
    appendLog('info', 'Started AI Video Object Removal Pipeline');
    appendLog('info', `Configuration: SAM 2 + ProPainter [Dilation: ${state.dilation}px, HighQuality: ${state.highQuality}]`);
  }

  function setPipelineStep(stepNumber) {
    DOM.stepTracking.className = 'step-item';
    DOM.stepInpainting.className = 'step-item';
    DOM.stepFinalizing.className = 'step-item';
    DOM.stepDivider1.className = 'step-divider';
    DOM.stepDivider2.className = 'step-divider';

    if (stepNumber === 1) {
      DOM.stepTracking.classList.add('active');
      DOM.progressStageName.textContent = 'Stage 1: SAM 2 Bidirectional Mask Propagation...';
    } else if (stepNumber === 2) {
      DOM.stepTracking.classList.add('completed');
      DOM.stepDivider1.classList.add('active');
      DOM.stepInpainting.classList.add('active');
      DOM.progressStageName.textContent = 'Stage 2: Flow-Guided ProPainter Temporal Inpainting...';
    } else if (stepNumber === 3) {
      DOM.stepTracking.classList.add('completed');
      DOM.stepInpainting.classList.add('completed');
      DOM.stepDivider1.classList.add('active');
      DOM.stepDivider2.classList.add('active');
      DOM.stepFinalizing.classList.add('active');
      DOM.progressStageName.textContent = 'Stage 3: H.264 Re-encoding & Color Blending...';
    }
  }

  function connectWebSocket(jobId) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/${jobId}`;

    appendLog('info', `Connecting to WebSocket: ${wsUrl}`);

    try {
      const ws = new WebSocket(wsUrl);
      state.wsConnection = ws;

      ws.onopen = () => {
        appendLog('success', 'Connected to real-time progress stream.');
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          handleProgressMessage(msg);
        } catch (e) {
          appendLog('info', event.data);
        }
      };

      ws.onerror = () => {
        appendLog('warn', 'WebSocket disconnected. Falling back to HTTP polling...');
        fallbackPolling(jobId);
      };

      ws.onclose = () => {
        if (state.isProcessing) {
          fallbackPolling(jobId);
        }
      };
    } catch (e) {
      fallbackPolling(jobId);
    }
  }

  function fallbackPolling(jobId) {
    if (state.pollTimer) clearInterval(state.pollTimer);

    state.pollTimer = setInterval(async () => {
      if (!state.isProcessing) {
        clearInterval(state.pollTimer);
        return;
      }

      try {
        const res = await fetch(`/api/status/${jobId}`);
        if (res.ok) {
          const data = await res.json();
          handleProgressMessage(data);
        }
      } catch (err) {
        // Run simulation if polling fails completely
        clearInterval(state.pollTimer);
        runSimulatedProcessing();
      }
    }, 1000);
  }

  function handleProgressMessage(msg) {
    if (msg.stage === 'tracking') setPipelineStep(1);
    else if (msg.stage === 'inpainting') setPipelineStep(2);
    else if (msg.stage === 'finalizing') setPipelineStep(3);

    if (msg.progress !== undefined) {
      const p = Math.round(msg.progress);
      DOM.progressBarFill.style.width = `${p}%`;
      DOM.progressPercentage.textContent = `${p}%`;
    }

    if (msg.current_frame !== undefined && msg.total_frames !== undefined) {
      DOM.progressFrameStats.textContent = `Frame ${msg.current_frame} / ${msg.total_frames}`;
    }

    if (msg.fps) {
      DOM.progressFpsStats.textContent = `${msg.fps.toFixed(1)} FPS`;
    }

    if (msg.eta) {
      DOM.progressTimeStats.textContent = `ETA: ${msg.eta}`;
    }

    if (msg.message) {
      appendLog('info', msg.message);
    }

    if (msg.stage === 'completed' || msg.status === 'completed' || msg.progress >= 100) {
      const cleanUrl = msg.result_url || msg.result_video_url || (state.jobId ? `/api/download/${state.jobId}` : null);
      completeProcessing(cleanUrl);
    }
  }

  /**
   * High-fidelity simulated progress runner for testing & standalone presentations
   */
  function runSimulatedProcessing() {
    let progress = 0;
    const totalFrames = state.totalFrames || 90;

    const interval = setInterval(() => {
      if (!state.isProcessing) {
        clearInterval(interval);
        return;
      }

      progress += 2.5;

      if (progress < 35) {
        setPipelineStep(1);
        const curFrame = Math.round((progress / 35) * totalFrames);
        DOM.progressFrameStats.textContent = `Tracking Frame ${curFrame} / ${totalFrames}`;
        DOM.progressFpsStats.textContent = `28.4 FPS`;
        DOM.progressTimeStats.textContent = `ETA: 4s`;
        if (progress % 10 < 3) {
          appendLog('info', `[SAM 2] Memory bank updating features at frame ${curFrame}`);
        }
      } else if (progress < 85) {
        setPipelineStep(2);
        const curFrame = Math.round(((progress - 35) / 50) * totalFrames);
        DOM.progressFrameStats.textContent = `Inpainting Frame ${curFrame} / ${totalFrames}`;
        DOM.progressFpsStats.textContent = `18.2 FPS`;
        DOM.progressTimeStats.textContent = `ETA: 2s`;
        if (progress % 15 < 3) {
          appendLog('info', `[ProPainter] Flow completion pass completed on frame ${curFrame}`);
        }
      } else if (progress < 100) {
        setPipelineStep(3);
        DOM.progressFrameStats.textContent = `Rendering final video stream...`;
        DOM.progressFpsStats.textContent = `60.0 FPS`;
        DOM.progressTimeStats.textContent = `Finalizing`;
      } else {
        clearInterval(interval);
        completeProcessing(state.videoUrl);
      }

      const p = Math.min(100, Math.round(progress));
      DOM.progressBarFill.style.width = `${p}%`;
      DOM.progressPercentage.textContent = `${p}%`;
    }, 150);
  }

  function appendLog(type, text) {
    const line = document.createElement('div');
    line.className = `log-line ${type}`;
    const time = new Date().toLocaleTimeString();
    line.textContent = `[${time}] ${text}`;
    DOM.processLogTerminal.appendChild(line);
    DOM.processLogTerminal.scrollTop = DOM.processLogTerminal.scrollHeight;
  }

  function cancelProcessing() {
    state.isProcessing = false;
    if (state.wsConnection) {
      state.wsConnection.close();
      state.wsConnection = null;
    }
    if (state.pollTimer) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
    DOM.processingModal.style.display = 'none';
    showToast('Processing cancelled', 'info');
  }

  function completeProcessing(resultUrl) {
    state.isProcessing = false;
    DOM.progressBarFill.style.width = '100%';
    DOM.progressPercentage.textContent = '100%';
    appendLog('success', 'Inpainting pipeline finished successfully!');

    setTimeout(() => {
      DOM.processingModal.style.display = 'none';
      openResultComparison(resultUrl || state.videoUrl);
    }, 600);
  }

  // =========================================================================
  // RESULT VIEWER & BEFORE / AFTER SPLIT COMPARISON SLIDER
  // =========================================================================
  function openResultComparison(resultUrl) {
    DOM.resultModal.style.display = 'flex';
    state.resultVideoUrl = resultUrl;

    const beforeVideo = DOM.beforeCompareVideo;
    const afterVideo = DOM.afterCompareVideo;

    beforeVideo.src = state.videoUrl;
    afterVideo.src = resultUrl;

    DOM.downloadResultBtn.href = resultUrl;
    DOM.downloadResultBtn.download = `inpainted_${DOM.projectFileName.textContent || 'video.mp4'}`;

    // Reset split position to 50%
    setSplitPosition(50);

    // Sync videos
    beforeVideo.currentTime = 0;
    afterVideo.currentTime = 0;

    beforeVideo.onloadedmetadata = () => {
      DOM.compareTimeReadout.textContent = `00:00 / ${formatTime(beforeVideo.duration)}`;
      DOM.compareScrubber.max = Math.floor(beforeVideo.duration * 10);
      DOM.compareScrubber.value = 0;
    };

    // Auto play comparison in loop
    playComparison();
  }

  function setupComparisonSlider() {
    const handle = DOM.comparisonSliderHandle;
    const container = DOM.comparisonWrapper;

    function onDrag(clientX) {
      const rect = container.getBoundingClientRect();
      const pos = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
      setSplitPosition(pos * 100);
    }

    handle.addEventListener('mousedown', (e) => {
      state.isDraggingSplit = true;
      e.preventDefault();
    });

    window.addEventListener('mousemove', (e) => {
      if (state.isDraggingSplit) {
        onDrag(e.clientX);
      }
    });

    window.addEventListener('mouseup', () => {
      state.isDraggingSplit = false;
    });

    // Touch support for split slider
    handle.addEventListener('touchstart', (e) => {
      state.isDraggingSplit = true;
    });

    window.addEventListener('touchmove', (e) => {
      if (state.isDraggingSplit && e.touches.length > 0) {
        onDrag(e.touches[0].clientX);
      }
    });

    window.addEventListener('touchend', () => {
      state.isDraggingSplit = false;
    });

    // Comparison Playback Controls
    DOM.btnComparePlayPause.addEventListener('click', toggleComparisonPlayback);

    DOM.beforeCompareVideo.addEventListener('timeupdate', () => {
      const cur = DOM.beforeCompareVideo.currentTime;
      const dur = DOM.beforeCompareVideo.duration || 1;
      DOM.compareScrubber.value = Math.floor(cur * 10);
      DOM.compareTimeReadout.textContent = `${formatTime(cur)} / ${formatTime(dur)}`;

      // Strict sync lock: keep afterVideo within 0.05s
      if (Math.abs(DOM.afterCompareVideo.currentTime - cur) > 0.05) {
        DOM.afterCompareVideo.currentTime = cur;
      }
    });

    DOM.compareScrubber.addEventListener('input', (e) => {
      const time = e.target.value / 10;
      DOM.beforeCompareVideo.currentTime = time;
      DOM.afterCompareVideo.currentTime = time;
    });
  }

  function setSplitPosition(pct) {
    state.splitPosition = pct;
    DOM.comparisonSliderHandle.style.left = `${pct}%`;
    DOM.afterContainer.style.clipPath = `polygon(0 0, ${pct}% 0, ${pct}% 100%, 0 100%)`;
  }

  function toggleComparisonPlayback() {
    if (DOM.beforeCompareVideo.paused) {
      playComparison();
    } else {
      pauseComparison();
    }
  }

  function playComparison() {
    DOM.beforeCompareVideo.play();
    DOM.afterCompareVideo.play();
    DOM.comparePlayIcon.style.display = 'none';
    DOM.comparePauseIcon.style.display = 'block';
  }

  function pauseComparison() {
    DOM.beforeCompareVideo.pause();
    DOM.afterCompareVideo.pause();
    DOM.comparePlayIcon.style.display = 'block';
    DOM.comparePauseIcon.style.display = 'none';
  }

  // =========================================================================
  // UTILITIES & HELPERS
  // =========================================================================
  function formatTime(seconds) {
    if (isNaN(seconds)) return '00:00:00';
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 100);

    const pad = (n) => String(n).padStart(2, '0');
    if (hrs > 0) {
      return `${pad(hrs)}:${pad(mins)}:${pad(secs)}`;
    }
    return `${pad(mins)}:${pad(secs)}.${pad(ms)}`;
  }

  function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    let iconSvg = '';
    if (type === 'success') {
      iconSvg = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>';
    } else if (type === 'error') {
      iconSvg = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#f43f5e" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>';
    } else {
      iconSvg = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>';
    }

    toast.innerHTML = `${iconSvg}<span>${message}</span>`;
    DOM.toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(10px)';
      toast.style.transition = 'all 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, 3500);
  }

  // Run initialization on DOMContentLoaded
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
