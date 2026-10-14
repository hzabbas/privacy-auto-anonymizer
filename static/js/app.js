/**
 * Privacy Auto-Anonymizer - client-side studio
 *
 * Architecture (100% client-side masking, the server only returns geometry):
 *  1. Clean export at native 1:1 resolution without any helper graphics.
 *  2. Full-auto redaction mode (one-click sanitisation of every detection).
 *  3. Before/After comparison tool (hold-to-peek + Spacebar).
 *  4. Human-in-the-loop toggling: DP-Pix for faces/plates, solid black for text.
 *
 * Scientific grounding (see RESEARCH_PAPERS.md):
 *  - ReGenHuman (2026): interactive human-in-the-loop curation.
 *  - Incremental Image Anonymization (2025): DP-Pix differential-privacy pixelation.
 *  - RedactionBench (2026): zero-entropy contextual-integrity masking for text.
 */

class AnonymizerStudio {
    constructor() {
        // DOM elements
        this.canvas = document.getElementById('editor-canvas') || document.getElementById('imageCanvas');
        this.canvasContainer = document.getElementById('canvasContainer');
        this.connectorLayer = document.getElementById('connector-layer');
        this.labelsLayer = document.getElementById('labels-layer');
        this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
        this.dropzone = document.getElementById('dropzoneContainer');
        this.placeholder = document.getElementById('placeholderState');
        this.uploadInputs = document.querySelectorAll('input[type="file"]');
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.statusBadge = document.getElementById('statusBadge');
        this.statusText = document.getElementById('statusText');
        this.legendContainer = document.getElementById('legendContainer');

        // Control buttons
        this.btnAutoRedact = document.getElementById('btnAutoRedact');
        this.btnSelectAll = document.getElementById('btnSelectAll');
        this.btnDeselectAll = document.getElementById('btnDeselectAll');
        this.btnCompare = document.getElementById('btnCompare');
        this.btnExport = document.getElementById('btnExport');

        // State
        this.originalImage = null;
        this.currentFile = null;
        this.currentObjectUrl = null;
        this.detections = [];            // [{ id, type, bbox, polygon, score, isMasked }]
        this.isProcessing = false;
        this.hoveredDetectionId = null;
        this.isComparing = false;
        this.requestSeq = 0;             // invalidates stale in-flight analyses
        this.pendingReanalysis = false;  // re-run requested while a request was active
        this.badgeEls = new Map();       // detection id -> HUD badge element
        this.toastTimeout = null;
        this.isDrawing = false;          // manual redaction box currently dragged
        this.drawStart = null;           // drag anchor in native image pixels
        this.drawCurrent = null;         // live drag corner in native image pixels
        this.suppressClickToggle = false;// the click after a drag must not toggle
        this.hudResizePending = false;   // rAF throttle for the ResizeObserver
        this.resizeObserver = null;

        // Entity colour themes
        this.colorConfig = {
            face: {
                stroke: '#06b6d4',      // neon cyan
                fill: 'rgba(6, 182, 212, 0.15)',
                label: 'چهره (Face)'
            },
            plate: {
                stroke: '#10b981',      // neon emerald
                fill: 'rgba(16, 185, 129, 0.15)',
                label: 'پلاک خودرو (Plate)'
            },
            text: {
                stroke: '#f59e0b',      // neon amber
                fill: 'rgba(245, 158, 11, 0.15)',
                label: 'متن حساس (Text)'
            },
            manual: {
                stroke: '#f43f5e',      // rose
                fill: 'rgba(244, 63, 94, 0.15)',
                label: 'ناحیه دستی (سفارشی)'
            },
            default: {
                stroke: '#8b5cf6',
                fill: 'rgba(139, 92, 246, 0.15)',
                label: 'عنصر حساس'
            }
        };

        // Persian labels for the entity names returned by the backend.
        this.entityLabelsFa = {
            'Credit card': 'کارت اعتباری',
            'IBAN': 'شبا (IBAN)',
            'SSN': 'تأمین اجتماعی',
            'Secret / key': 'رمز یا کلید',
            'Email': 'ایمیل',
            'IP address': 'نشانی IP',
            'Phone': 'تلفن',
            'ID number': 'کد شناسایی',
            'Date of birth': 'تاریخ تولد',
            'Date': 'تاریخ',
            'MRZ': 'نوار خوانش ماشینی (MRZ)',
            'Address': 'نشانی',
            'URL': 'نشانی وب',
            'Person name': 'نام شخص',
            'Location': 'مکان'
        };

        this.initEventListeners();
    }

    // ------------------------------------------------------------------ //
    // Small helpers                                                       //
    // ------------------------------------------------------------------ //

    /** Escapes text before it is interpolated into an HTML template. */
    escapeHtml(value) {
        if (value === null || value === undefined) return '';
        return String(value).replace(/[&<>"']/g, (c) => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
        }[c]));
    }

    /** Reads the Django CSRF cookie so the API POST is accepted. */
    getCsrfToken() {
        const cookies = document.cookie ? document.cookie.split(';') : [];
        for (const cookie of cookies) {
            const parts = cookie.trim().split('=');
            if (parts[0] === 'csrftoken') {
                return decodeURIComponent(parts.slice(1).join('='));
            }
        }
        return '';
    }

    isFormElement(element) {
        const tag = ((element && element.tagName) || '').toLowerCase();
        return ['input', 'select', 'textarea', 'button', 'a'].includes(tag) ||
            Boolean(element && element.isContentEditable);
    }

    /** Translates viewport coordinates to native canvas pixels. */
    getCanvasCoordinates(event) {
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;
        return {
            x: (event.clientX - rect.left) * scaleX,
            y: (event.clientY - rect.top) * scaleY
        };
    }

    isPointInPolygon(point, polygon) {
        let inside = false;
        for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
            const xi = polygon[i][0], yi = polygon[i][1];
            const xj = polygon[j][0], yj = polygon[j][1];
            const intersect = ((yi > point.y) !== (yj > point.y)) &&
                (point.x < (xj - xi) * (point.y - yi) / (yj - yi) + xi);
            if (intersect) inside = !inside;
        }
        return inside;
    }

    /** Hit testing: returns the smallest detection under the point. */
    findHitDetection(x, y) {
        if (this.isComparing) return null;

        const hits = this.detections.filter(det => {
            if (det.polygon && Array.isArray(det.polygon) && det.polygon.length >= 3) {
                return this.isPointInPolygon({ x, y }, det.polygon);
            }
            const [x1, y1, x2, y2] = det.bbox;
            return x >= x1 && x <= x2 && y >= y1 && y <= y2;
        });

        if (hits.length === 0) return null;

        hits.sort((a, b) => {
            const areaA = (a.bbox[2] - a.bbox[0]) * (a.bbox[3] - a.bbox[1]);
            const areaB = (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]);
            return areaA - areaB;
        });
        return hits[0];
    }

    // ------------------------------------------------------------------ //
    // Event wiring                                                        //
    // ------------------------------------------------------------------ //

    initEventListeners() {
        this.uploadInputs.forEach(input => {
            input.addEventListener('change', (event) => {
                const file = event.target.files && event.target.files[0];
                if (file) this.loadFile(file);
                event.target.value = '';   // allow re-selecting the same file
            });
        });

        if (this.dropzone) {
            ['dragenter', 'dragover'].forEach(eventName => {
                this.dropzone.addEventListener(eventName, (event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    this.dropzone.classList.add('drag-active');
                });
            });

            ['dragleave', 'drop'].forEach(eventName => {
                this.dropzone.addEventListener(eventName, (event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    this.dropzone.classList.remove('drag-active');
                });
            });

            this.dropzone.addEventListener('drop', (event) => {
                const dt = event.dataTransfer;
                if (dt && dt.files && dt.files[0]) {
                    this.loadFile(dt.files[0]);
                }
            });
        }

        // Engine / reasoning toggles -> automatic re-analysis
        const ocrSelect = document.getElementById('ocrEngineSelect');
        const layoutSelect = document.getElementById('layoutEngineSelect');
        const imajevToggle = document.getElementById('imajevToggle');
        const reanalyzeHandler = () => {
            if (this.currentFile) this.analyzeImage(this.currentFile);
        };
        if (ocrSelect) ocrSelect.addEventListener('change', reanalyzeHandler);
        if (layoutSelect) layoutSelect.addEventListener('change', reanalyzeHandler);
        if (imajevToggle) {
            imajevToggle.addEventListener('change', (event) => {
                this.handleImajevToggle(event.target.checked);
                reanalyzeHandler();
            });
        }

        const btnClearLogs = document.getElementById('btnClearImajevLogs');
        if (btnClearLogs) {
            btnClearLogs.addEventListener('click', () => {
                const content = document.getElementById('imajevLogContent');
                const countBadge = document.getElementById('imajevLogCount');
                if (content) {
                    content.innerHTML = '<div class="text-zinc-500 text-[11px] italic p-2.5 bg-zinc-900/30 rounded-lg border border-zinc-800/40">لاگ‌ها پاکسازی شدند.</div>';
                }
                if (countBadge) countBadge.textContent = '0 رویداد';
            });
        }

        // Canvas interaction (human-in-the-loop): click toggles a mask, and a
        // click-and-drag on empty space draws a custom redaction region.
        this.canvas.addEventListener('click', (event) => this.handleCanvasClick(event));
        this.canvas.addEventListener('mousedown', (event) => this.handleCanvasMouseDown(event));
        this.canvas.addEventListener('mousemove', (event) => this.handleCanvasMouseMove(event));
        // The drag keeps tracking even when the pointer leaves the canvas.
        window.addEventListener('mousemove', (event) => this.handleDrawMove(event));
        window.addEventListener('mouseup', (event) => this.handleDrawEnd(event));
        this.canvas.addEventListener('mouseleave', () => {
            if (this.isDrawing) return;
            if (this.hoveredDetectionId !== null) {
                this.hoveredDetectionId = null;
                this.canvas.style.cursor = '';
                this.render();
            }
        });

        // The workspace animates (grid toggle, panel open/close, window resize),
        // which changes the canvas' CSS size over 300ms. HUD callouts and SVG
        // connectors are positioned from getBoundingClientRect(), so they are
        // recomputed during the transition - not only when it ends.
        if (typeof ResizeObserver !== 'undefined') {
            this.resizeObserver = new ResizeObserver(() => this.scheduleHudRealign());
            if (this.canvasContainer) this.resizeObserver.observe(this.canvasContainer);
            this.resizeObserver.observe(this.canvas);
        }

        // Batch actions
        if (this.btnAutoRedact) {
            this.btnAutoRedact.addEventListener('click', () => {
                if (this.detections.length === 0) return;
                this.detections.forEach(det => { det.isMasked = true; });
                this.updateStatusSummary();
                this.render();
                this.showToast('تمام عناصر حساس سانسور شدند.');
            });
        }

        if (this.btnSelectAll) {
            this.btnSelectAll.addEventListener('click', () => {
                if (this.detections.length === 0) return;
                this.detections.forEach(det => { det.isMasked = true; });
                this.updateStatusSummary();
                this.render();
            });
        }

        if (this.btnDeselectAll) {
            this.btnDeselectAll.addEventListener('click', () => {
                if (this.detections.length === 0) return;
                this.detections.forEach(det => { det.isMasked = false; });
                this.updateStatusSummary();
                this.render();
            });
        }

        // Before/After comparison (hold to peek)
        if (this.btnCompare) {
            const startCompare = (event) => {
                event.preventDefault();
                if (!this.originalImage) return;
                this.setComparing(true);
            };
            const stopCompare = (event) => {
                event.preventDefault();
                if (!this.originalImage) return;
                this.setComparing(false);
            };

            this.btnCompare.addEventListener('mousedown', startCompare);
            this.btnCompare.addEventListener('mouseup', stopCompare);
            this.btnCompare.addEventListener('mouseleave', stopCompare);
            this.btnCompare.addEventListener('blur', stopCompare);
            this.btnCompare.addEventListener('touchstart', startCompare, { passive: false });
            this.btnCompare.addEventListener('touchend', stopCompare, { passive: false });
            this.btnCompare.addEventListener('touchcancel', stopCompare, { passive: false });
            this.btnCompare.addEventListener('contextmenu', (event) => event.preventDefault());
        }

        // Spacebar peek (ignored while a form control has focus)
        window.addEventListener('keydown', (event) => {
            if (event.code !== 'Space' || this.isComparing || !this.originalImage) return;
            if (this.isFormElement(event.target)) return;
            event.preventDefault();
            this.setComparing(true);
        });
        window.addEventListener('keyup', (event) => {
            if (event.code === 'Space' && this.isComparing) {
                event.preventDefault();
                this.setComparing(false);
            }
        });

        // Responsive HUD re-layout (debounced through requestAnimationFrame)
        let resizeFrame = null;
        window.addEventListener('resize', () => {
            if (!this.originalImage) return;
            if (resizeFrame) cancelAnimationFrame(resizeFrame);
            resizeFrame = requestAnimationFrame(() => {
                resizeFrame = null;
                this.render();
            });
        });

        if (this.btnExport) {
            this.btnExport.addEventListener('click', () => this.exportCleanImage());
        }
    }

    setComparing(active) {
        if (this.isComparing === active) return;
        this.isComparing = active;
        if (active) {
            this.hoveredDetectionId = null;
            this.canvas.style.cursor = '';
            this.canvas.classList.remove('draw-ready');
        } else if (this.originalImage) {
            this.canvas.classList.add('draw-ready');
        }
        this.render();
    }

    handleCanvasClick(event) {
        if (this.suppressClickToggle) {
            // The mouseup that ended a drag gesture must not also toggle a box.
            this.suppressClickToggle = false;
            return;
        }
        if (!this.originalImage || this.detections.length === 0 || this.isComparing) return;
        const { x, y } = this.getCanvasCoordinates(event);
        const hit = this.findHitDetection(x, y);
        if (!hit) return;
        hit.isMasked = !hit.isMasked;
        this.updateStatusSummary();
        this.render();
    }

    handleCanvasMouseMove(event) {
        if (!this.originalImage || this.isComparing || this.isDrawing) return;
        const { x, y } = this.getCanvasCoordinates(event);
        const hit = this.findHitDetection(x, y);

        if (hit) {
            this.canvas.style.cursor = 'pointer';
            if (this.hoveredDetectionId !== hit.id) {
                this.hoveredDetectionId = hit.id;
                this.render();
            }
        } else {
            this.canvas.style.cursor = '';   // falls back to the crosshair class
            if (this.hoveredDetectionId !== null) {
                this.hoveredDetectionId = null;
                this.render();
            }
        }
    }

    // ------------------------------------------------------------------ //
    // Manual redaction regions (click-and-drag on empty canvas space)     //
    // ------------------------------------------------------------------ //

    /** Arms the drawing gesture when the press lands on empty canvas space. */
    handleCanvasMouseDown(event) {
        if (!this.originalImage || this.isComparing || event.button !== 0) return;
        const { x, y } = this.getCanvasCoordinates(event);
        // A press on an existing polygon stays a click (mask toggle).
        if (this.findHitDetection(x, y)) return;
        this.drawStart = this.clampToCanvas(x, y);
        this.drawCurrent = this.drawStart;
        this.isDrawing = false;          // true once the pointer really moves
        this.suppressClickToggle = false;
    }

    handleDrawMove(event) {
        if (!this.drawStart || this.isComparing) return;
        const { x, y } = this.getCanvasCoordinates(event);
        const current = this.clampToCanvas(x, y);
        if (!this.isDrawing) {
            const travel = Math.hypot(current.x - this.drawStart.x, current.y - this.drawStart.y);
            if (travel < 6) return;      // still a click, not a drag yet
            this.isDrawing = true;
            this.canvas.style.cursor = 'crosshair';
        }
        this.drawCurrent = current;
        this.render();
    }

    handleDrawEnd() {
        if (!this.drawStart) return;
        const wasDrawing = this.isDrawing;
        const rect = wasDrawing ? this.getDrawRect() : null;
        this.drawStart = null;
        this.drawCurrent = null;
        this.isDrawing = false;
        if (!wasDrawing) return;

        this.suppressClickToggle = true;   // consume the click fired after mouseup
        this.canvas.style.cursor = '';

        const MANUAL_MIN_SIZE = 12;        // native-image pixels for a usable region
        const width = rect[2] - rect[0];
        const height = rect[3] - rect[1];
        if (width >= MANUAL_MIN_SIZE && height >= MANUAL_MIN_SIZE) {
            const [x1, y1, x2, y2] = rect;
            this.detections.push({
                id: this.nextDetectionId(),
                type: 'manual',
                entity: 'CUSTOM',
                label: 'ناحیه دستی (سفارشی)',
                score: 1.0,
                bbox: [x1, y1, x2, y2],
                polygon: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                // isMasked is the flag the renderer / exporter read; redacted
                // mirrors it for readers that use that descriptive name.
                isMasked: true,
                redacted: true,
                isManual: true
            });
            this.updateStatusSummary();
            this.showToast('ناحیه دستی اضافه شد؛ برای حذف از × روی برچسب استفاده کنید.');
        }
        this.render();
    }

    /** Live dashed cyan/amber rectangle while a manual region is dragged. */
    drawSelectionPreview(ctx) {
        const [x1, y1, x2, y2] = this.getDrawRect();
        const width = x2 - x1;
        const height = y2 - y1;
        if (width <= 0 || height <= 0) return;
        ctx.save();
        ctx.setLineDash([8, 6]);
        ctx.lineWidth = 2;
        ctx.strokeStyle = 'rgba(34, 211, 238, 0.95)';
        ctx.fillStyle = 'rgba(34, 211, 238, 0.14)';
        ctx.fillRect(x1, y1, width, height);
        ctx.strokeRect(x1, y1, width, height);
        ctx.setLineDash([4, 4]);
        ctx.lineWidth = 1;
        ctx.strokeStyle = 'rgba(245, 158, 11, 0.9)';
        ctx.strokeRect(x1 + 2, y1 + 2, Math.max(0, width - 4), Math.max(0, height - 4));
        ctx.restore();
    }

    /** Normalised [x1, y1, x2, y2] of the box currently being dragged. */
    getDrawRect() {
        const anchor = this.drawStart || { x: 0, y: 0 };
        const corner = this.drawCurrent || anchor;
        return [
            Math.round(Math.min(anchor.x, corner.x)),
            Math.round(Math.min(anchor.y, corner.y)),
            Math.round(Math.max(anchor.x, corner.x)),
            Math.round(Math.max(anchor.y, corner.y))
        ];
    }

    clampToCanvas(x, y) {
        return {
            x: Math.max(0, Math.min(x, this.canvas.width)),
            y: Math.max(0, Math.min(y, this.canvas.height))
        };
    }

    /** Id for a manual detection: one above the highest current id. */
    nextDetectionId() {
        return this.detections.reduce((highest, det) => Math.max(highest, det.id || 0), 0) + 1;
    }

    /** Removes a manual region (its badge, its hover state and its mask). */
    removeDetection(detectionId) {
        const before = this.detections.length;
        this.detections = this.detections.filter(det => det.id !== detectionId);
        if (this.detections.length === before) return;
        const badge = this.badgeEls.get(detectionId);
        if (badge) {
            badge.remove();
            this.badgeEls.delete(detectionId);
        }
        if (this.hoveredDetectionId === detectionId) this.hoveredDetectionId = null;
        this.updateStatusSummary();
        this.render();
        this.showToast('ناحیه دستی حذف شد.');
    }

    /** Drops any in-flight drawing state (new image, comparison mode). */
    cancelDrawing() {
        this.isDrawing = false;
        this.drawStart = null;
        this.drawCurrent = null;
        this.suppressClickToggle = false;
    }

    /** rAF-throttled HUD realign, driven by the ResizeObserver. */
    scheduleHudRealign() {
        if (!this.originalImage || this.hudResizePending) return;
        this.hudResizePending = true;
        requestAnimationFrame(() => {
            this.hudResizePending = false;
            if (this.originalImage && !this.isComparing) this.render();
        });
    }

    // ------------------------------------------------------------------ //
    // File loading + analysis                                             //
    // ------------------------------------------------------------------ //

    loadFile(file) {
        if (!file) return;

        const name = file.name || '';
        const dotIndex = name.lastIndexOf('.');
        const extension = dotIndex >= 0 ? name.slice(dotIndex + 1).toLowerCase() : '';
        const looksLikeImage = (file.type && file.type.startsWith('image/')) ||
            ['jpg', 'jpeg', 'png', 'webp', 'bmp', 'gif', 'tif', 'tiff'].includes(extension);
        if (!looksLikeImage) {
            this.showToast('لطفاً یک فایل تصویری معتبر (JPG، PNG یا WEBP) انتخاب کنید.', true);
            return;
        }

        // Any analysis still in flight belongs to the previous image.
        this.requestSeq += 1;
        this.pendingReanalysis = false;
        this.currentFile = file;
        this.detections = [];
        this.hoveredDetectionId = null;
        this.cancelDrawing();
        this.clearCalloutLayers();
        this.enableControls(false);
        if (this.legendContainer) {
            this.legendContainer.classList.add('hidden');
            this.legendContainer.classList.remove('flex');
        }
        if (this.statusBadge) this.statusBadge.classList.add('hidden');
        if (this.dropzone) this.dropzone.style.minHeight = '';
        if (this.canvasContainer) this.canvasContainer.style.maxWidth = '';

        if (this.currentObjectUrl) {
            URL.revokeObjectURL(this.currentObjectUrl);
            this.currentObjectUrl = null;
        }
        const objectUrl = URL.createObjectURL(file);
        this.currentObjectUrl = objectUrl;

        const img = new Image();
        img.onload = () => {
            this.originalImage = img;

            // Canvas runs at the native image resolution; CSS scales it for display.
            this.canvas.width = img.naturalWidth;
            this.canvas.height = img.naturalHeight;

            if (this.placeholder) this.placeholder.classList.add('hidden');
            if (this.canvasContainer) this.canvasContainer.classList.remove('hidden');
            this.canvas.classList.remove('hidden');
            this.canvas.classList.add('draw-ready');   // enables manual region drawing
            if (this.dropzone) {
                this.dropzone.classList.remove('border-dashed');
                this.dropzone.classList.add('border-solid');
            }

            this.render();
            this.analyzeImage(file);
        };
        img.onerror = () => {
            this.showToast('فایل تصویری قابل خواندن نیست.', true);
        };
        img.src = objectUrl;
    }

    /** True while the response still belongs to the newest request for the current image. */
    isLatestRequest(seq, file) {
        return seq === this.requestSeq && file === this.currentFile;
    }

    async analyzeImage(file) {
        if (!file) return;

        if (this.isProcessing) {
            // A request is already running; run this one right after it finishes.
            this.pendingReanalysis = true;
            return;
        }

        const seq = ++this.requestSeq;
        const targetFile = file;
        this.setLoading(true);
        if (this.statusBadge) this.statusBadge.classList.add('hidden');

        try {
            const formData = new FormData();
            formData.append('image', targetFile);
            formData.append('conf_threshold', '0.35');
            formData.append('ocr_engine', document.getElementById('ocrEngineSelect')?.value || 'easyocr');
            formData.append('layout_engine', document.getElementById('layoutEngineSelect')?.value || 'regex');

            const imajevToggle = document.getElementById('imajevToggle');
            const useImajev = imajevToggle ? imajevToggle.checked : false;
            formData.append('use_imajev', useImajev ? 'true' : 'false');
            if (useImajev) this.setImajevBusyState();

            const response = await fetch('/api/analyze/', {
                method: 'POST',
                body: formData,
                headers: { 'X-CSRFToken': this.getCsrfToken() }
            });

            if (!this.isLatestRequest(seq, targetFile)) return;

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.message || ('خطای سرور: ' + response.status));
            }

            const data = await response.json();
            if (!this.isLatestRequest(seq, targetFile)) return;
            this.applyAnalysisResult(data, useImajev);
        } catch (error) {
            if (!this.isLatestRequest(seq, targetFile)) return;
            console.error('[Anonymizer] Error during image analysis:', error);
            this.detections = [];
            this.clearCalloutLayers();
            this.render();
            if (this.statusBadge) this.statusBadge.classList.add('hidden');
            this.showToast('خطا: ' + error.message, true);
        } finally {
            // Only one request can be in flight (guarded above), so clearing the
            // overlay here is always safe - even when this response was stale.
            this.setLoading(false);
            if (this.pendingReanalysis && this.currentFile) {
                this.pendingReanalysis = false;
                this.analyzeImage(this.currentFile);
            }
        }
    }

    applyAnalysisResult(data, useImajev) {
        if (data.status !== 'success' || !Array.isArray(data.detections)) {
            throw new Error(data.message || 'پاسخ نامعتبر از سرور دریافت شد.');
        }

        // Manual regions survive a re-analysis (engine switch, Imajev toggle);
        // they get fresh ids after the server detections so the two id spaces
        // can never collide.
        const manualDetections = this.detections.filter(det => det.isManual);
        this.detections = data.detections.map(det => ({ ...det, isMasked: false }));
        manualDetections.forEach(det => {
            det.id = this.nextDetectionId();
            this.detections.push(det);
        });

        if (useImajev && Array.isArray(data.imajev_logs)) {
            this.renderImajevLogs(data.imajev_logs);
        } else if (!useImajev) {
            const statusText = document.getElementById('imajevStatusText');
            if (statusText) statusText.textContent = 'غیرفعال';
        }

        this.enableControls(true);
        this.updateStatusSummary();
        this.render();
    }

    setLoading(loading) {
        this.isProcessing = loading;
        if (!this.loadingOverlay) return;

        const textElement = this.loadingOverlay.querySelector('p');
        if (loading) {
            this.loadingOverlay.classList.remove('hidden');
            this.loadingOverlay.classList.add('flex');

            if (textElement) {
                textElement.style.transition = 'opacity 0.3s ease';
                textElement.style.opacity = '1';
                textElement.textContent = 'در حال پردازش تصویر...';

                if (this.loadingTimeout) clearTimeout(this.loadingTimeout);
                this.loadingTimeout = setTimeout(() => {
                    if (!this.isProcessing) return;
                    textElement.style.opacity = '0';
                    setTimeout(() => {
                        textElement.textContent = 'در حال بارگذاری یا دانلود مدل‌های هوشمند (در اولین اجرا ممکن است چند دقیقه زمان ببرد)...';
                        textElement.style.opacity = '1';
                    }, 300);
                }, 6000);
            }
        } else {
            this.loadingOverlay.classList.remove('flex');
            this.loadingOverlay.classList.add('hidden');
            if (this.loadingTimeout) {
                clearTimeout(this.loadingTimeout);
                this.loadingTimeout = null;
            }
        }
    }

    enableControls(enabled) {
        [this.btnAutoRedact, this.btnSelectAll, this.btnDeselectAll, this.btnCompare, this.btnExport]
            .forEach(btn => { if (btn) btn.disabled = !enabled; });

        if (this.legendContainer) {
            const show = enabled && this.detections.length > 0;
            this.legendContainer.classList.toggle('hidden', !show);
            this.legendContainer.classList.toggle('flex', show);
        }
    }

    updateStatusSummary() {
        if (!this.statusBadge || !this.statusText) return;

        const total = this.detections.length;
        const dot = this.statusBadge.querySelector('span.w-2');

        if (total === 0) {
            this.statusText.textContent = 'هیچ عنصر حساسی یافت نشد';
            if (dot) {
                dot.classList.remove('bg-emerald-500');
                dot.classList.add('bg-amber-500');
            }
        } else {
            const maskedCount = this.detections.filter(det => det.isMasked).length;
            this.statusText.textContent = total + ' مورد شناسایی شد (' + maskedCount + ' ماسک فعال)';
            if (dot) {
                dot.classList.remove('bg-amber-500');
                dot.classList.add('bg-emerald-500');
            }
        }

        if (this.legendContainer) {
            this.legendContainer.classList.toggle('hidden', total === 0);
            this.legendContainer.classList.toggle('flex', total > 0);
        }

        this.statusBadge.classList.remove('hidden');
        this.statusBadge.classList.add('flex');
    }

    setImajevBusyState() {
        const statusText = document.getElementById('imajevStatusText');
        const statusPulse = document.getElementById('imajevStatusPulse');
        if (statusText) statusText.textContent = 'در حال استدلال...';
        if (statusPulse) {
            statusPulse.classList.remove('bg-emerald-400');
            statusPulse.classList.add('bg-cyan-400');
        }
        const container = document.getElementById('imajevLogContent');
        if (container) {
            container.innerHTML = '<div class="text-cyan-400 text-[11px] font-mono flex items-center gap-2 p-2.5 bg-zinc-900/50 rounded-lg border border-zinc-800/60 animate-pulse">' +
                '<span class="w-2 h-2 rounded-full bg-cyan-400 animate-ping"></span>' +
                'در حال ارسال تصویر و اجرای استدلال معنایی توسط Imajev VLM...</div>';
        }
    }

    showToast(message, isError = false) {
        let toast = document.getElementById('app-toast');
        if (!toast) {
            toast = document.createElement('div');
            toast.id = 'app-toast';
            toast.className = 'fixed bottom-5 left-1/2 -translate-x-1/2 z-[100] max-w-[92vw] px-4 py-2.5 rounded-lg shadow-xl text-xs font-medium backdrop-blur-md border transition-all duration-300 toast-hidden';
            const icon = document.createElement('span');
            icon.className = 'inline-block w-2 h-2 rounded-full mr-2 align-middle';
            const text = document.createElement('span');
            text.className = 'align-middle';
            toast.appendChild(icon);
            toast.appendChild(text);
            document.body.appendChild(toast);
        }

        const icon = toast.firstElementChild;
        const text = toast.lastElementChild;

        toast.classList.remove('border-emerald-500/40', 'bg-zinc-900/95', 'text-emerald-300',
            'border-red-500/50', 'bg-red-950/90', 'text-red-100');
        if (isError) {
            toast.classList.add('border-red-500/50', 'bg-red-950/90', 'text-red-100');
            icon.className = 'inline-block w-2 h-2 rounded-full mr-2 align-middle bg-red-400';
        } else {
            toast.classList.add('border-emerald-500/40', 'bg-zinc-900/95', 'text-emerald-300');
            icon.className = 'inline-block w-2 h-2 rounded-full mr-2 align-middle bg-emerald-400';
        }
        // textContent (never innerHTML): server messages can never inject markup.
        text.textContent = message;

        toast.classList.remove('toast-hidden');
        if (this.toastTimeout) clearTimeout(this.toastTimeout);
        this.toastTimeout = setTimeout(() => toast.classList.add('toast-hidden'), 3800);
    }

    // ------------------------------------------------------------------ //
    // Canvas rendering                                                    //
    // ------------------------------------------------------------------ //

    render() {
        if (!this.originalImage) return;

        const { width, height } = this.canvas;
        const ctx = this.ctx;

        ctx.clearRect(0, 0, width, height);
        ctx.drawImage(this.originalImage, 0, 0, width, height);

        if (this.isComparing) {
            this.clearCalloutLayers();
            this.drawComparisonWatermark(ctx, width, height);
            return;
        }

        this.detections.forEach(det => {
            if (det.isMasked) this.applyClientMask(ctx, det, width, height);
        });

        this.detections.forEach(det => {
            const isHovered = (this.hoveredDetectionId === det.id);
            if (det.isMasked) {
                this.drawMaskedOverlay(ctx, det, isHovered);
            } else {
                this.drawSuggestionBox(ctx, det, isHovered);
            }
        });

        // Live dashed cyan/amber rectangle while a manual region is dragged.
        if (this.isDrawing) this.drawSelectionPreview(ctx);

        this.renderHUDCallouts();
    }

    drawComparisonWatermark(ctx, width, height) {
        ctx.save();
        const text = 'تصویر اصلی خام (قبل از پالایش)';
        const fontSize = Math.max(13, Math.min(18, Math.round(width / 50)));
        ctx.font = '600 ' + fontSize + 'px Inter, Vazirmatn, sans-serif';

        const paddingX = 14;
        const paddingY = 8;
        const metrics = ctx.measureText(text);
        const badgeW = metrics.width + paddingX * 2;
        const badgeH = fontSize + paddingY * 2;

        const x = (width - badgeW) / 2;
        const y = 20;

        ctx.fillStyle = 'rgba(9, 9, 11, 0.85)';
        ctx.shadowColor = 'rgba(0, 0, 0, 0.6)';
        ctx.shadowBlur = 10;
        this.roundRect(ctx, x, y, badgeW, badgeH, 6);
        ctx.fill();

        ctx.strokeStyle = 'rgba(6, 182, 212, 0.8)';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        ctx.fillStyle = '#ffffff';
        ctx.shadowBlur = 0;
        ctx.textBaseline = 'middle';
        ctx.fillText(text, x + paddingX, y + (badgeH / 2));
        ctx.restore();
    }

    /**
     * Client-side redaction filters:
     *  - text / manual: zero-entropy solid black following the oriented polygon.
     *  - face/plate: DP-Pix (mosaic + additive noise, Incremental Anonymization 2025).
     */
    applyClientMask(ctx, det, canvasW, canvasH) {
        let [x1, y1, x2, y2] = det.bbox;
        x1 = Math.max(0, Math.min(x1, canvasW));
        y1 = Math.max(0, Math.min(y1, canvasH));
        x2 = Math.max(0, Math.min(x2, canvasW));
        y2 = Math.max(0, Math.min(y2, canvasH));

        const w = x2 - x1;
        const h = y2 - y1;
        if (w <= 0 || h <= 0) return;

        const hasPolygon = det.polygon && Array.isArray(det.polygon) && det.polygon.length >= 3;

        if (det.type === 'text' || det.type === 'manual') {
            ctx.save();
            try {
                ctx.fillStyle = '#000000';
                if (hasPolygon) {
                    ctx.beginPath();
                    ctx.moveTo(det.polygon[0][0], det.polygon[0][1]);
                    for (let i = 1; i < det.polygon.length; i++) {
                        ctx.lineTo(det.polygon[i][0], det.polygon[i][1]);
                    }
                    ctx.closePath();
                    ctx.fill();
                } else {
                    ctx.fillRect(x1, y1, w, h);
                }
            } finally {
                ctx.restore();
            }
            return;
        }

        ctx.save();
        try {
            if (hasPolygon) {
                ctx.beginPath();
                ctx.moveTo(det.polygon[0][0], det.polygon[0][1]);
                for (let i = 1; i < det.polygon.length; i++) {
                    ctx.lineTo(det.polygon[i][0], det.polygon[i][1]);
                }
                ctx.closePath();
                ctx.clip();
            }

            const imgData = ctx.getImageData(x1, y1, w, h);
            const data = imgData.data;

            // Mosaic cell size scales with the region so small faces still lose detail.
            const blockSize = Math.max(8, Math.min(22, Math.round(Math.min(w, h) / 7)));
            const noiseScale = 12.0;

            for (let by = 0; by < h; by += blockSize) {
                for (let bx = 0; bx < w; bx += blockSize) {
                    const bw = Math.min(blockSize, w - bx);
                    const bh = Math.min(blockSize, h - by);

                    let rSum = 0, gSum = 0, bSum = 0, count = 0;
                    for (let dy = 0; dy < bh; dy++) {
                        for (let dx = 0; dx < bw; dx++) {
                            const idx = ((by + dy) * w + (bx + dx)) * 4;
                            rSum += data[idx];
                            gSum += data[idx + 1];
                            bSum += data[idx + 2];
                            count++;
                        }
                    }

                    // Additive stochastic perturbation (DP-Pix formulation).
                    const noiseR = (Math.random() - 0.5) * 2 * noiseScale;
                    const noiseG = (Math.random() - 0.5) * 2 * noiseScale;
                    const noiseB = (Math.random() - 0.5) * 2 * noiseScale;

                    const avgR = Math.min(255, Math.max(0, Math.round(rSum / count + noiseR)));
                    const avgG = Math.min(255, Math.max(0, Math.round(gSum / count + noiseG)));
                    const avgB = Math.min(255, Math.max(0, Math.round(bSum / count + noiseB)));

                    for (let dy = 0; dy < bh; dy++) {
                        for (let dx = 0; dx < bw; dx++) {
                            const idx = ((by + dy) * w + (bx + dx)) * 4;
                            data[idx] = avgR;
                            data[idx + 1] = avgG;
                            data[idx + 2] = avgB;
                        }
                    }
                }
            }

            ctx.putImageData(imgData, x1, y1);
        } catch (err) {
            // getImageData can fail on very large canvases in memory-limited browsers.
            console.warn('[Anonymizer] Falling back to a solid mask:', err);
            ctx.fillStyle = '#18181b';
            ctx.fillRect(x1, y1, w, h);
        } finally {
            ctx.restore();
        }
    }

    /** Unmasked suggestion box with neon styling (tilted polygon or rectangle). */
    drawSuggestionBox(ctx, det, isHovered) {
        const config = this.colorConfig[det.type] || this.colorConfig.default;
        ctx.save();

        const hasPolygon = det.polygon && Array.isArray(det.polygon) && det.polygon.length >= 3;
        if (hasPolygon) {
            ctx.beginPath();
            ctx.moveTo(det.polygon[0][0], det.polygon[0][1]);
            for (let i = 1; i < det.polygon.length; i++) {
                ctx.lineTo(det.polygon[i][0], det.polygon[i][1]);
            }
            ctx.closePath();

            ctx.fillStyle = isHovered ? config.fill.replace('0.15', '0.28') : config.fill;
            ctx.fill();

            ctx.strokeStyle = config.stroke;
            ctx.lineWidth = isHovered ? 3.0 : 2.0;
            ctx.shadowColor = config.stroke;
            ctx.shadowBlur = isHovered ? 10 : 5;
            ctx.stroke();
        } else {
            const [x1, y1, x2, y2] = det.bbox;
            const boxWidth = x2 - x1;
            const boxHeight = y2 - y1;
            if (boxWidth <= 0 || boxHeight <= 0) {
                ctx.restore();
                return;
            }

            ctx.fillStyle = isHovered ? config.fill.replace('0.15', '0.28') : config.fill;
            ctx.fillRect(x1, y1, boxWidth, boxHeight);

            ctx.strokeStyle = config.stroke;
            ctx.lineWidth = isHovered ? 3.0 : 2.0;
            ctx.shadowColor = config.stroke;
            ctx.shadowBlur = isHovered ? 10 : 5;
            ctx.strokeRect(x1, y1, boxWidth, boxHeight);

            this.drawCornerAccents(ctx, x1, y1, boxWidth, boxHeight, config.stroke);
        }

        ctx.restore();
    }

    /** Masked area outline (no text on the canvas). */
    drawMaskedOverlay(ctx, det, isHovered) {
        ctx.save();
        ctx.strokeStyle = isHovered ? 'rgba(239, 68, 68, 0.9)' : 'rgba(16, 185, 129, 0.7)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash(isHovered ? [4, 4] : []);

        if (det.polygon && Array.isArray(det.polygon) && det.polygon.length >= 3) {
            ctx.beginPath();
            ctx.moveTo(det.polygon[0][0], det.polygon[0][1]);
            for (let i = 1; i < det.polygon.length; i++) {
                ctx.lineTo(det.polygon[i][0], det.polygon[i][1]);
            }
            ctx.closePath();
            ctx.stroke();
        } else {
            const [x1, y1, x2, y2] = det.bbox;
            const boxWidth = x2 - x1;
            const boxHeight = y2 - y1;
            if (boxWidth <= 0 || boxHeight <= 0) {
                ctx.restore();
                return;
            }
            ctx.strokeRect(x1, y1, boxWidth, boxHeight);
        }

        ctx.restore();
    }

    drawCornerAccents(ctx, x, y, w, h, color) {
        const length = Math.min(10, w / 4, h / 4);
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.shadowBlur = 0;

        ctx.beginPath();
        ctx.moveTo(x, y + length);
        ctx.lineTo(x, y);
        ctx.lineTo(x + length, y);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(x + w - length, y);
        ctx.lineTo(x + w, y);
        ctx.lineTo(x + w, y + length);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(x, y + h - length);
        ctx.lineTo(x, y + h);
        ctx.lineTo(x + length, y + h);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(x + w - length, y + h);
        ctx.lineTo(x + w, y + h);
        ctx.lineTo(x + w, y + h - length);
        ctx.stroke();

        ctx.restore();
    }

    roundRect(ctx, x, y, width, height, radius) {
        if (ctx.roundRect) {
            ctx.beginPath();
            ctx.roundRect(x, y, width, height, radius);
            return;
        }
        ctx.beginPath();
        ctx.moveTo(x + radius, y);
        ctx.lineTo(x + width - radius, y);
        ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
        ctx.lineTo(x + width, y + height - radius);
        ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
        ctx.lineTo(x + radius, y + height);
        ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
        ctx.lineTo(x, y + radius);
        ctx.quadraticCurveTo(x, y, x + radius, y);
        ctx.closePath();
    }

    clearCalloutLayers() {
        if (this.connectorLayer) this.connectorLayer.textContent = '';
        if (this.labelsLayer) this.labelsLayer.textContent = '';
        this.badgeEls.clear();
    }

    // ------------------------------------------------------------------ //
    // HUD callouts (SVG leaders + HTML badges)                            //
    // ------------------------------------------------------------------ //

    /** Label + tooltip for a detection (Persian for the backend entity names). */
    getDisplayLabel(det) {
        const config = this.colorConfig[det.type] || this.colorConfig.default;
        let label = config.label;
        let title = det.label ? (config.label + ' - ' + det.label) : config.label;

        if (det.type === 'text') {
            const faLabel = det.label ? (this.entityLabelsFa[det.label] || det.label) : '';
            label = faLabel ? ('متن: ' + faLabel) : 'متن حساس';
            if (det.label) title = 'متن حساس: ' + faLabel + ' (' + det.label + ')';
        } else if (det.type === 'plate') {
            label = 'پلاک خودرو';
        } else if (det.type === 'face') {
            label = 'چهره';
        } else if (det.type === 'manual') {
            label = 'ناحیه دستی';
            title = 'ناحیه دستی (سفارشی) — ناحیه‌ای که خودتان رسم کرده‌اید (برای حذف × را بزنید)';
        }
        return { label, title };
    }

    /**
     * Renders the HUD callouts.
     *
     * Geometry: when the dropzone is wide enough the badges sit in lateral
     * gutters beside the image (leaders point at the box edge); on narrow
     * screens they move inside the image edges instead of being clipped. The
     * badge width adapts to the longest label currently on screen.
     *
     * Stability: connector lines are rebuilt on every render (they carry no
     * state), but each badge element is created once per detection and then
     * updated in place - hovering must never destroy the node under the cursor.
     */
    renderHUDCallouts() {
        if (!this.connectorLayer || !this.labelsLayer) return;

        if (this.detections.length === 0) {
            this.clearCalloutLayers();
            if (this.dropzone) this.dropzone.style.minHeight = '';
            return;
        }

        // Width fits the longest visible label (padding + score chip included).
        let badgeWidth = 134;
        this.detections.forEach(det => {
            const { label } = this.getDisplayLabel(det);
            badgeWidth = Math.max(badgeWidth, Math.min(196, Math.round(58 + label.length * 7.2)));
        });
        const BADGE_WIDTH = badgeWidth;
        const GAP_FROM_IMAGE = 14;
        const GUTTER = BADGE_WIDTH + GAP_FROM_IMAGE;

        // Reserve lateral room for the badges whenever the dropzone allows it.
        const availableWidth = this.dropzone ? this.dropzone.clientWidth : this.canvas.clientWidth;
        const canUseGutters = availableWidth >= (GUTTER * 2) + 340;
        if (this.canvasContainer) {
            this.canvasContainer.style.maxWidth = canUseGutters ? 'calc(100% - ' + (GUTTER * 2) + 'px)' : '100%';
        }

        const rect = this.canvas.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return;

        const scaleX = rect.width / this.canvas.width;
        const scaleY = rect.height / this.canvas.height;

        const items = this.detections.map(det => {
            const [x1, y1, x2, y2] = det.bbox;
            const sx1 = x1 * scaleX;
            const sy1 = y1 * scaleY;
            const sx2 = x2 * scaleX;
            const sy2 = y2 * scaleY;
            const boxCenterX = (sx1 + sx2) / 2;
            return {
                det,
                sx1, sy1, sx2, sy2,
                boxCenterX,
                boxCenterY: (sy1 + sy2) / 2,
                // The gutter side follows the *centre* of the box: a detection
                // whose centre lies in the left half of the image goes to the
                // left gutter, the right half to the right gutter. All values
                // are physical pixels from getBoundingClientRect, so the choice
                // is independent of the document's RTL direction; the leader
                // line then attaches to the nearest vertical edge (sx1 / sx2).
                isLeftSide: boxCenterX <= rect.width / 2,
                targetY: (sy1 + sy2) / 2
            };
        });

        const leftItems = items.filter(item => item.isLeftSide);
        const rightItems = items.filter(item => !item.isLeftSide);
        const totalCount = Math.max(leftItems.length, rightItems.length);

        // Vertical headroom: the badges may use the space above/below the image
        // (inside the dropzone) and the dropzone grows when a column gets busy.
        const dropzoneRect = this.dropzone ? this.dropzone.getBoundingClientRect() : rect;
        const spaceTop = Math.max(0, rect.top - dropzoneRect.top);
        const spaceBottom = Math.max(0, dropzoneRect.bottom - rect.bottom);

        if (this.dropzone) {
            if (totalCount >= 7) {
                const desiredMinHeight = Math.max(540, totalCount * 46 + 60);
                this.dropzone.style.minHeight = desiredMinHeight + 'px';
            } else {
                this.dropzone.style.minHeight = '';
            }
        }

        const topLimit = -Math.min(Math.max(spaceTop - 16, 0), 50);
        const bottomLimit = rect.height + Math.min(Math.max(spaceBottom - 16, 0), 60);
        const usableHeight = Math.max(120, bottomLimit - topLimit);
        const desiredSpacing = Math.min(46, Math.max(34, (usableHeight - 20) / Math.max(totalCount, 1)));
        const maxBottom = bottomLimit - 10;

        // Classic label stacking: keep each badge near its box, push down on
        // collision, then shift the whole column back up if it overflows.
        const placeLabels = (list) => {
            if (list.length === 0) return;
            list.sort((a, b) => a.boxCenterY - b.boxCenterY);

            let cursor = -Infinity;
            list.forEach(item => {
                let y = Math.max(item.boxCenterY, topLimit);
                if (y < cursor) y = cursor;
                item.targetY = y;
                cursor = y + desiredSpacing;
            });

            const last = list[list.length - 1];
            const overflow = last.targetY - maxBottom;
            if (overflow > 0) {
                const shift = Math.min(overflow, Math.max(0, list[0].targetY - topLimit));
                if (shift > 0) list.forEach(item => { item.targetY -= shift; });
            }

            if (list[list.length - 1].targetY > maxBottom + 0.5) {
                const step = (maxBottom - topLimit) / Math.max(list.length - 1, 1);
                list.forEach((item, i) => { item.targetY = topLimit + i * step; });
            }
        };

        placeLabels(leftItems);
        placeLabels(rightItems);

        const svgFragment = document.createDocumentFragment();
        const activeIds = new Set();

        items.forEach(item => {
            const { det, sx1, sx2, boxCenterY, isLeftSide } = item;
            const config = this.colorConfig[det.type] || this.colorConfig.default;
            const isHovered = (this.hoveredDetectionId === det.id);
            const isMasked = det.isMasked;
            const strokeColor = isMasked ? (isHovered ? '#ef4444' : '#10b981') : config.stroke;
            const clampedY = Math.min(Math.max(item.targetY, topLimit), bottomLimit - 10);

            // Badge + leader geometry
            let p0, p1, p2, p3, badgeLeftPx;
            if (canUseGutters) {
                if (isLeftSide) {
                    badgeLeftPx = -GUTTER;
                    p0 = { x: sx1, y: boxCenterY };
                    p1 = { x: Math.max(sx1 - 10, -4), y: boxCenterY };
                    p2 = { x: -8, y: clampedY };
                    p3 = { x: -GAP_FROM_IMAGE, y: clampedY };
                } else {
                    badgeLeftPx = rect.width + GAP_FROM_IMAGE;
                    p0 = { x: sx2, y: boxCenterY };
                    p1 = { x: Math.min(sx2 + 10, rect.width + 4), y: boxCenterY };
                    p2 = { x: rect.width + 8, y: clampedY };
                    p3 = { x: rect.width + GAP_FROM_IMAGE, y: clampedY };
                }
            } else {
                // Inside mode: badges hug the image edges so nothing is clipped.
                if (isLeftSide) {
                    badgeLeftPx = 8;
                    const attachX = badgeLeftPx + BADGE_WIDTH;
                    p0 = { x: sx1, y: boxCenterY };
                    p1 = { x: Math.max(sx1 - 8, 0), y: boxCenterY };
                    p2 = { x: Math.min(attachX + 12, rect.width), y: clampedY };
                    p3 = { x: attachX, y: clampedY };
                } else {
                    badgeLeftPx = rect.width - BADGE_WIDTH - 8;
                    const attachX = badgeLeftPx;
                    p0 = { x: sx2, y: boxCenterY };
                    p1 = { x: Math.min(sx2 + 8, rect.width), y: boxCenterY };
                    p2 = { x: Math.max(attachX - 12, 0), y: clampedY };
                    p3 = { x: attachX, y: clampedY };
                }
            }

            // A. SVG connector
            const groupEl = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            groupEl.setAttribute('class', 'hud-connector');

            const dot0 = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            dot0.setAttribute('cx', p0.x.toFixed(1));
            dot0.setAttribute('cy', p0.y.toFixed(1));
            dot0.setAttribute('r', isHovered ? '4' : '3');
            dot0.setAttribute('fill', strokeColor);
            groupEl.appendChild(dot0);

            if (isHovered) {
                const dotGlow = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                dotGlow.setAttribute('cx', p0.x.toFixed(1));
                dotGlow.setAttribute('cy', p0.y.toFixed(1));
                dotGlow.setAttribute('r', '7');
                dotGlow.setAttribute('fill', strokeColor);
                dotGlow.setAttribute('opacity', '0.25');
                groupEl.appendChild(dotGlow);
            }

            const polyline = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
            polyline.setAttribute('points',
                p0.x.toFixed(1) + ',' + p0.y.toFixed(1) + ' ' +
                p1.x.toFixed(1) + ',' + p1.y.toFixed(1) + ' ' +
                p2.x.toFixed(1) + ',' + p2.y.toFixed(1) + ' ' +
                p3.x.toFixed(1) + ',' + p3.y.toFixed(1));
            polyline.setAttribute('stroke', strokeColor);
            polyline.setAttribute('stroke-width', isHovered ? '2.2' : '1.5');
            polyline.setAttribute('stroke-linecap', 'round');
            polyline.setAttribute('stroke-linejoin', 'round');
            polyline.setAttribute('fill', 'none');
            polyline.setAttribute('opacity', isHovered ? '1' : (isMasked ? '0.8' : '0.85'));
            if (!isMasked && !isHovered) {
                polyline.setAttribute('stroke-dasharray', '4,3');
            }
            groupEl.appendChild(polyline);
            svgFragment.appendChild(groupEl);

            // B. HTML badge (created once, updated in place)
            activeIds.add(det.id);
            const badge = this.ensureBadge(det.id);
            this.updateBadge(badge, item, { badgeLeftPx, clampedY, strokeColor, isHovered, isMasked, config, badgeWidth: BADGE_WIDTH });
        });

        this.connectorLayer.textContent = '';
        this.connectorLayer.appendChild(svgFragment);

        // Drop badges whose detection no longer exists.
        Array.from(this.badgeEls.keys()).forEach(id => {
            if (!activeIds.has(id)) {
                const el = this.badgeEls.get(id);
                if (el) el.remove();
                this.badgeEls.delete(id);
            }
        });
    }

    ensureBadge(detectionId) {
        let el = this.badgeEls.get(detectionId);
        if (el) return el;

        el = document.createElement('div');
        el.dataset.detectionId = String(detectionId);
        el.setAttribute('role', 'button');
        el.setAttribute('tabindex', '0');
        el.style.position = 'absolute';
        el.style.transform = 'translate(0, -50%)';

        const toggle = (event) => {
            if (event) {
                event.stopPropagation();
                event.preventDefault();
            }
            const detection = this.detections.find(item => item.id === detectionId);
            if (!detection) return;
            // The little × on a manual region removes it instead of toggling.
            if (event && event.target && event.target.closest &&
                    event.target.closest('[data-remove-detection]')) {
                this.removeDetection(detectionId);
                return;
            }
            detection.isMasked = !detection.isMasked;
            this.updateStatusSummary();
            this.render();
        };

        el.addEventListener('click', toggle);
        el.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' || event.key === ' ') toggle(event);
        });
        el.addEventListener('mouseenter', () => {
            if (this.hoveredDetectionId !== detectionId) {
                this.hoveredDetectionId = detectionId;
                this.render();
            }
        });
        el.addEventListener('mouseleave', () => {
            if (this.hoveredDetectionId === detectionId) {
                this.hoveredDetectionId = null;
                this.render();
            }
        });

        this.labelsLayer.appendChild(el);
        this.badgeEls.set(detectionId, el);
        return el;
    }

    updateBadge(el, item, geometry) {
        const { det } = item;
        const { badgeLeftPx, clampedY, strokeColor, isHovered, isMasked, badgeWidth } = geometry;

        el.style.top = clampedY.toFixed(1) + 'px';
        el.style.left = badgeLeftPx.toFixed(1) + 'px';
        el.style.width = badgeWidth + 'px';

        const borderCol = isMasked
            ? (isHovered ? 'border-red-500/90 shadow-[0_0_15px_rgba(239,68,68,0.4)]' : 'border-emerald-500/80 shadow-[0_0_12px_rgba(16,185,129,0.3)]')
            : (isHovered ? 'border-cyan-400 shadow-[0_0_15px_rgba(6,182,212,0.4)]' : 'border-zinc-800/90 shadow-[0_4px_16px_rgba(0,0,0,0.6)]');
        const bgCol = isHovered ? 'bg-[#141418] text-white scale-[1.04]' : 'bg-[#0a0a0e]/95 text-zinc-200';
        const className = 'hud-badge pointer-events-auto cursor-pointer select-none px-2 py-1 rounded-xl border ' +
            borderCol + ' ' + bgCol +
            ' text-xs font-medium flex items-center justify-between gap-1.5 backdrop-blur-md whitespace-nowrap z-20 hover:scale-105 active:scale-95 group';
        if (el.dataset.cls !== className) {
            el.className = className;
            el.dataset.cls = className;
        }

        const { label: displayLabel, title: fullTitle } = this.getDisplayLabel(det);
        const scorePercent = Math.round((det.score || 0) * 100);

        // Rebuild the inner markup only when its content actually changed; this
        // keeps a hovered badge alive while the canvas/HUD re-renders.
        const signature = [isMasked, scorePercent, displayLabel, fullTitle, det.isManual ? 'manual' : ''].join('|');
        if (el.dataset.signature !== signature) {
            el.dataset.signature = signature;
            const scoreChip = '<span class="text-[9px] font-mono font-semibold px-1.5 py-0.5 rounded-md bg-zinc-900/90 text-zinc-300 border border-zinc-700/60 flex-shrink-0">' + scorePercent + '%</span>';
            const labelSpan = '<span class="text-zinc-200 font-medium text-[10px] tracking-tight truncate" title="' + this.escapeHtml(fullTitle) + '">' + this.escapeHtml(displayLabel) + '</span>';
            const dotSpan = '<span class="w-1.5 h-1.5 rounded-full flex-shrink-0 transition-transform duration-200 group-hover:scale-125" style="background-color: ' + strokeColor + '; box-shadow: 0 0 6px ' + strokeColor + ';"></span>';
            const removeBtn = det.isManual
                ? '<button type="button" data-remove-detection="1" title="حذف ناحیه دستی" aria-label="حذف ناحیه دستی" class="hud-remove-btn">×</button>'
                : '';
            el.innerHTML = item.isLeftSide
                ? scoreChip + '<div class="flex items-center gap-1.5 overflow-hidden">' + labelSpan + dotSpan + '</div>' + removeBtn
                : '<div class="flex items-center gap-1.5 overflow-hidden">' + dotSpan + labelSpan + '</div>' + scoreChip + removeBtn;
        }

        el.title = fullTitle;
        el.setAttribute('aria-pressed', isMasked ? 'true' : 'false');
        el.setAttribute('aria-label', displayLabel + ' - اطمینان ' + scorePercent + '٪ - ' + (isMasked ? 'ماسک فعال (برای لغو کلیک کنید)' : 'بدون ماسک (برای فعال‌سازی کلیک کنید)'));
    }

    // ------------------------------------------------------------------ //
    // Export                                                             //
    // ------------------------------------------------------------------ //

    /**
     * Renders the sanitised image on an off-screen canvas at native 1:1
     * resolution - masks only, never helper graphics - and triggers a download.
     */
    exportCleanImage() {
        if (!this.originalImage) {
            this.showToast('تصویری برای استخراج وجود ندارد.', true);
            return;
        }

        const nativeWidth = this.originalImage.naturalWidth;
        const nativeHeight = this.originalImage.naturalHeight;

        const offCanvas = document.createElement('canvas');
        offCanvas.width = nativeWidth;
        offCanvas.height = nativeHeight;
        const offCtx = offCanvas.getContext('2d', { willReadFrequently: true });

        offCtx.drawImage(this.originalImage, 0, 0, nativeWidth, nativeHeight);
        this.detections.forEach(det => {
            if (det.isMasked) this.applyClientMask(offCtx, det, nativeWidth, nativeHeight);
        });

        const maskedCount = this.detections.filter(det => det.isMasked).length;
        const originalName = (this.currentFile && this.currentFile.name) || '';
        const dotIndex = originalName.lastIndexOf('.');
        const baseName = dotIndex > 0 ? originalName.slice(0, dotIndex) : (originalName || 'image');

        offCanvas.toBlob((blob) => {
            if (!blob) {
                this.showToast('ساخت فایل خروجی ناموفق بود.', true);
                return;
            }
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.style.display = 'none';
            link.href = url;
            link.download = 'redacted_' + baseName + '.png';
            document.body.appendChild(link);
            link.click();
            setTimeout(() => {
                link.remove();
                URL.revokeObjectURL(url);
            }, 1000);

            if (maskedCount > 0) {
                this.showToast(maskedCount + ' ناحیه پالایش شد و تصویر دانلود گردید.');
            } else {
                this.showToast('هیچ ناحیه‌ای ماسک نشده بود؛ تصویر اصلی دانلود شد.', true);
            }
        }, 'image/png');
    }

    // ------------------------------------------------------------------ //
    // Imajev panel                                                        //
    // ------------------------------------------------------------------ //

    /**
     * Shows/hides the Imajev terminal panel and switches the workspace between
     * the single-column and the desktop two-column layout (canvas + 340px
     * terminal). The base grid-cols-1 always stays so mobile remains stacked.
     */
    setImajevPanelVisible(visible) {
        const workspaceGrid = document.getElementById('workspaceGrid');
        const dropzoneContainer = document.getElementById('dropzoneContainer');
        const logPanel = document.getElementById('imajevLogPanel');

        if (workspaceGrid) {
            workspaceGrid.classList.toggle('lg:grid-cols-[1fr_340px]', visible);
            workspaceGrid.classList.toggle('imajev-grid-split', visible);
            workspaceGrid.classList.add('grid-cols-1');
            workspaceGrid.classList.remove('lg:grid-cols-4');
        }
        if (dropzoneContainer) {
            // min-w-0 lets the canvas column shrink inside the grid instead of
            // overflowing when the terminal takes its fixed column.
            dropzoneContainer.classList.toggle('min-w-0', visible);
            dropzoneContainer.classList.remove('lg:col-span-3');
        }
        if (logPanel) {
            logPanel.classList.toggle('hidden', !visible);
            logPanel.classList.toggle('flex', visible);
            logPanel.classList.remove('lg:col-span-1');
        }
    }

    /** Toggles the two-column workspace layout when Imajev is enabled/disabled. */
    handleImajevToggle(isActive) {
        const mainWorkspace = document.getElementById('mainWorkspace');
        const statusText = document.getElementById('imajevStatusText');

        this.setImajevPanelVisible(isActive);

        if (mainWorkspace) {
            mainWorkspace.classList.toggle('max-w-7xl', !isActive);
            mainWorkspace.classList.toggle('max-w-[1560px]', isActive);
        }
        if (isActive) {
            this.showToast('حالت استدلال هوشمند Imajev فعال شد.');
        } else if (statusText) {
            statusText.textContent = 'غیرفعال';
        }

        // Let the layout animation settle, then refresh the HUD alignment.
        setTimeout(() => {
            if (this.originalImage) this.render();
        }, 160);
    }

    /**
     * Splits a leading "emoji Imajev:" / "Imajev/OCR:" prefix off a log message.
     *
     * The prefix is pure Latin text; leaving it inside the Persian sentence made
     * the browser's bidi algorithm treat the whole line as LTR (the trailing
     * period jumped to the right, parentheses flipped). The prefix is rendered
     * as its own dir="ltr" badge and the sentence keeps its own direction.
     */
    splitLogPrefix(message) {
        const match = /^\s*([✨✅❌⚠🚀ℹ⚡🔍🛡🔒🌐\uFE0F]*)\s*(Imajev(?:\/OCR)?)\s*[:：]\s*/.exec(message || '');
        if (!match) {
            return { badge: '', body: String(message || '') };
        }
        return {
            badge: (match[1] ? match[1] + ' ' : '') + match[2] + ':',
            body: String(message).slice(match[0].length)
        };
    }

    /**
     * Paragraph direction taken from the first strong directional character:
     * a Persian sentence gets dir="rtl", a pure-English one keeps dir="ltr".
     * (dir="auto" on the raw message latched onto the Latin "Imajev:" prefix
     * and scrambled everything that followed.)
     */
    detectTextDirection(value) {
        for (const char of String(value || '')) {
            if (/[\u0591-\u07FF\uFB1D-\uFDFD\uFE70-\uFEFC\u0600-\u06FF]/.test(char)) return 'rtl';
            if (/[A-Za-z]/.test(char)) return 'ltr';
        }
        return 'rtl';
    }

    /**
     * Escapes a log message and isolates its technical runs with <bdi> so the
     * bidi algorithm can never re-order them: parenthesised groups such as
     * "(اطمینان 99.9٪)" and the inline tokens OCR / MRZ / PII / YOLO.
     */
    formatLogMessageHtml(message) {
        let html = this.escapeHtml(message);
        html = html.replace(/\(([^()]{1,90})\)/g, '<bdi>($1)</bdi>');
        html = html.replace(/\b(OCR|MRZ|PII|YOLO|VLM|PDF417)\b/g, '<bdi>$1</bdi>');
        html = html.replace(/(\d+(?:[.,]\d+)?[ \t]*[%٪])/g, '<bdi>$1</bdi>');
        return html;
    }

    renderImajevLogs(logs) {
        const container = document.getElementById('imajevLogContent');
        const countBadge = document.getElementById('imajevLogCount');
        const statusText = document.getElementById('imajevStatusText');
        const statusPulse = document.getElementById('imajevStatusPulse');
        if (!container) return;

        // A response that carries reasoning logs must leave the terminal panel
        // visible even when the layout update has not run yet.
        this.setImajevPanelVisible(true);

        container.textContent = '';
        if (countBadge) countBadge.textContent = (logs ? logs.length : 0) + ' رویداد';
        if (statusText) statusText.textContent = 'تکمیل شد';
        if (statusPulse) {
            statusPulse.classList.remove('bg-cyan-400');
            statusPulse.classList.add('bg-emerald-400');
        }

        if (!logs || logs.length === 0) {
            const empty = document.createElement('div');
            empty.className = 'text-zinc-500 text-[11px] italic p-3 bg-zinc-950/40 rounded-xl border border-zinc-800/40';
            empty.textContent = 'هیچ تصمیمی توسط سامانه استدلال Imajev برای این تصویر ثبت نشد.';
            container.appendChild(empty);
            return;
        }

        logs.forEach((logStr, index) => {
            let logData = {};
            if (typeof logStr === 'string') {
                try {
                    logData = JSON.parse(logStr);
                } catch (err) {
                    logData = { entity: 'system', message: logStr, action: 'info' };
                }
            } else if (typeof logStr === 'object' && logStr !== null) {
                logData = logStr;
            }

            const item = document.createElement('div');
            const action = String(logData.action || 'info').toLowerCase();

            let actionBadge = '';
            let itemBorder = 'border-zinc-800/80 hover:border-cyan-500/50 bg-zinc-900/80';
            let barColor = 'bg-cyan-500 shadow-[0_0_6px_rgba(6,182,212,0.5)]';
            let barDotColor = 'bg-cyan-400';

            if (action === 'approved') {
                actionBadge = '<span class="px-2 py-0.5 rounded text-[10px] font-bold tracking-wider font-mono bg-emerald-950/90 text-emerald-400 border border-emerald-500/80 shadow-[0_0_10px_rgba(16,185,129,0.35)] flex items-center gap-1 select-none"><span class="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>[APPROVED]</span>';
                itemBorder = 'border-emerald-900/70 hover:border-emerald-500/70 bg-zinc-950/90 shadow-[0_0_15px_rgba(16,185,129,0.06)]';
                barColor = 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.7)]';
                barDotColor = 'bg-emerald-400';
            } else if (action === 'rejected') {
                actionBadge = '<span class="px-2 py-0.5 rounded text-[10px] font-bold tracking-wider font-mono bg-red-950/90 text-red-400 border border-red-500/80 shadow-[0_0_10px_rgba(239,68,68,0.35)] flex items-center gap-1 select-none"><span class="w-1.5 h-1.5 rounded-full bg-red-400"></span>[REJECTED]</span>';
                itemBorder = 'border-red-900/70 hover:border-red-500/70 bg-zinc-950/90 shadow-[0_0_15px_rgba(239,68,68,0.06)]';
                barColor = 'bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.7)]';
                barDotColor = 'bg-red-400';
            } else if (action === 'error') {
                actionBadge = '<span class="px-2 py-0.5 rounded text-[10px] font-bold tracking-wider font-mono bg-amber-950/90 text-amber-400 border border-amber-600/80 shadow-[0_0_8px_rgba(245,158,11,0.3)] flex items-center gap-1 select-none">[ERROR]</span>';
                itemBorder = 'border-amber-900/70 hover:border-amber-500/70 bg-zinc-950/90';
                barColor = 'bg-amber-500 shadow-[0_0_6px_rgba(245,158,11,0.5)]';
                barDotColor = 'bg-amber-400';
            } else {
                actionBadge = '<span class="px-2 py-0.5 rounded text-[10px] font-bold tracking-wider font-mono bg-cyan-950/90 text-cyan-400 border border-cyan-800/80 shadow-[0_0_8px_rgba(6,182,212,0.25)] select-none">[INFO]</span>';
            }

            item.className = 'imajev-log-item group flex flex-col gap-2 p-3 rounded-xl border transition-all duration-200 relative overflow-hidden cursor-pointer shadow-sm ' + itemBorder;
            item.style.animationDelay = Math.min(index * 60, 600) + 'ms';

            if (logData.id) {
                item.dataset.detectionId = String(logData.id);
                item.title = 'برای فعال/غیرفعال کردن ماسک این عنصر کلیک کنید';
                item.addEventListener('mouseenter', () => {
                    if (this.hoveredDetectionId !== logData.id) {
                        this.hoveredDetectionId = logData.id;
                        this.render();
                    }
                });
                item.addEventListener('mouseleave', () => {
                    if (this.hoveredDetectionId === logData.id) {
                        this.hoveredDetectionId = null;
                        this.render();
                    }
                });
                item.addEventListener('click', () => {
                    const detection = this.detections.find(entry => entry.id === logData.id);
                    if (!detection) return;
                    detection.isMasked = !detection.isMasked;
                    this.updateStatusSummary();
                    this.render();
                });
            }

            let entityName = logData.entity ? String(logData.entity).toUpperCase() : 'SYSTEM';
            if (logData.text_preview) entityName += ' "' + logData.text_preview + '"';

            let messageMarkup = '';
            const message = logData.message || (logData.reason ? ('دلیل تصمیم: ' + logData.reason) : '');
            if (message) {
                // Bidi-safe rendering: the Latin "Imajev:" prefix becomes its own
                // dir="ltr" badge and the sentence itself gets an explicit
                // dir="rtl"/"ltr" from its first strong character, so a Persian
                // sentence can never be re-ordered as if it were LTR
                // (".فعال است OCR" style corruption).
                const parts = this.splitLogPrefix(message);
                const direction = this.detectTextDirection(parts.body);
                const alignClass = direction === 'rtl' ? 'text-right' : 'text-left';
                const badgeMarkup = parts.badge
                    ? '<span dir="ltr" class="inline-flex items-center font-mono text-[10px] font-semibold text-cyan-400/90 select-none align-middle" style="margin-inline-end: 0.375rem;">' + this.escapeHtml(parts.badge) + '</span>'
                    : '';
                messageMarkup = '<div class="imajev-log-message text-[11px] text-zinc-300 leading-relaxed font-sans px-0.5 ' + alignClass + '" dir="' + direction + '">' +
                    badgeMarkup + this.formatLogMessageHtml(parts.body) + '</div>';
            }

            let metricsMarkup = '';
            if (logData.confidence !== undefined && logData.confidence !== null) {
                const confPercent = Math.min(100, Math.max(0, Math.round(parseFloat(logData.confidence) * 100)));
                const unknownPercent = (logData.unknown_prob !== undefined && logData.unknown_prob !== null)
                    ? Math.min(100, Math.max(0, Math.round(parseFloat(logData.unknown_prob) * 100)))
                    : 0;

                let scoresMarkup = '';
                if (logData.scores && typeof logData.scores === 'object') {
                    const chips = Object.entries(logData.scores).map(([optionName, optionScore]) => {
                        const percent = Math.round(parseFloat(optionScore) * 100);
                        return '<span class="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-zinc-950/80 border border-zinc-800/80 text-[9px] text-zinc-400">' +
                            '<span class="text-zinc-300 truncate max-w-[170px]">' + this.escapeHtml(optionName) + ':</span>' +
                            '<span class="text-cyan-400 font-mono font-semibold">' + percent + '%</span></span>';
                    }).join('');
                    if (chips) {
                        scoresMarkup = '<div class="flex flex-wrap gap-1 pt-1.5 border-t border-zinc-800/40">' + chips + '</div>';
                    }
                }

                metricsMarkup =
                    '<div class="flex flex-col gap-1.5 pt-2 border-t border-zinc-800/60 text-[10px] font-mono select-none" dir="ltr">' +
                    '<div class="flex items-center gap-2">' +
                    '<span class="text-zinc-400 w-16 flex-shrink-0 flex items-center gap-1"><span class="w-1.5 h-1.5 rounded-full ' + barDotColor + '"></span>Target</span>' +
                    '<div class="flex-1 bg-zinc-950 rounded-full h-1.5 overflow-hidden border border-zinc-800/80"><div class="' + barColor + ' h-full rounded-full transition-all duration-700 ease-out" style="width: ' + confPercent + '%"></div></div>' +
                    '<span class="text-zinc-100 font-bold w-9 text-right">' + confPercent + '%</span>' +
                    '</div>' +
                    '<div class="flex items-center gap-2">' +
                    '<span class="text-zinc-400 w-16 flex-shrink-0 flex items-center gap-1"><span class="w-1.5 h-1.5 rounded-full ' + (unknownPercent > 40 ? 'bg-red-400 animate-pulse' : 'bg-amber-400') + '"></span>Unknown</span>' +
                    '<div class="flex-1 bg-zinc-950 rounded-full h-1.5 overflow-hidden border border-zinc-800/80"><div class="' + (unknownPercent > 40 ? 'bg-red-500 shadow-[0_0_6px_rgba(239,68,68,0.5)]' : 'bg-amber-400') + ' h-full rounded-full transition-all duration-700 ease-out" style="width: ' + unknownPercent + '%"></div></div>' +
                    '<span class="' + (unknownPercent > 40 ? 'text-red-400 font-bold' : 'text-amber-400/90') + ' font-semibold w-9 text-right">' + unknownPercent + '%</span>' +
                    '</div>' +
                    scoresMarkup +
                    '</div>';
            }

            item.innerHTML =
                '<div class="flex items-center justify-between gap-2" dir="ltr">' +
                '<div class="flex items-center gap-1.5 overflow-hidden">' +
                actionBadge +
                '<span class="text-[10px] font-semibold text-zinc-300 bg-zinc-800/90 border border-zinc-700/60 px-1.5 py-0.5 rounded tracking-wider uppercase truncate">' + this.escapeHtml(entityName) + '</span>' +
                '</div>' +
                '<span class="text-[9px] font-mono text-zinc-500 bg-zinc-950/60 px-1.5 py-0.5 rounded border border-zinc-800 flex-shrink-0">#' + String(index + 1).padStart(2, '0') + '</span>' +
                '</div>' +
                messageMarkup +
                metricsMarkup;

            container.appendChild(item);
        });

        container.scrollTop = container.scrollHeight;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    try {
        window.anonymizerStudio = new AnonymizerStudio();
    } catch (error) {
        console.error('[Anonymizer] Failed to initialise the studio:', error);
    }
});
