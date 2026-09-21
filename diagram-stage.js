(function () {
  const VIEWER_SRC = 'https://viewer.diagrams.net/js/viewer-static.min.js';

  function loadViewer() {
    if (window.__dsViewerPromise) return window.__dsViewerPromise;
    window.__dsViewerPromise = new Promise(function (resolve, reject) {
      if (window.GraphViewer) return resolve();
      const s = document.createElement('script');
      s.src = VIEWER_SRC;
      s.onload = function () { resolve(); };
      s.onerror = function () { reject(new Error('viewer script failed')); };
      document.head.appendChild(s);
    });
    return window.__dsViewerPromise;
  }

  class DiagramStage extends HTMLElement {
    static get observedAttributes() { return ['data-step', 'data-region-gap', 'data-block-gap', 'data-fade']; }

    constructor() {
      super();
      this.revealed = new Set();
      this.index = 0;
      this.timer = null;
      this.booted = false;
    }

    attr(name) {
      const v = this.getAttribute('data-' + name);
      return v === null ? this.getAttribute(name) : v;
    }

    num(name, fallback) {
      const v = parseFloat(this.attr(name));
      return isFinite(v) ? v : fallback;
    }

    get steps() { return (window[this.attr('timeline') || 'DIAGRAM_TIMELINE'] || []); }

    connectedCallback() {
      if (this.booted) return;
      this.booted = true;
      this.style.display = 'block';
      this.style.position = 'relative';
      this.style.overflow = 'hidden';
      if (!this.style.width) this.style.width = '100%';
      if (!this.style.height) this.style.height = '100%';
      this.host = document.createElement('div');
      this.host.style.cssText = 'position:absolute;inset:0;opacity:0;transition:opacity .25s linear';
      this.appendChild(this.host);
      this.boot().catch(function (e) { console.error('[diagram-stage]', e); });
    }

    attributeChangedCallback() { /* picked up on the next scheduled gap */ }

    async boot() {
      await loadViewer();
      /* offline bundles pre-register stencil sets that would otherwise be fetched */
      if (typeof window.__preloadStencils === 'function') {
        try { await window.__preloadStencils(); } catch (e) { console.warn('[diagram-stage] stencils', e); }
      }
      const rid = this.attr('resource-id') || 'diagramXml';
      const src = (window.__resources && window.__resources[rid]) || this.attr('src');
      let xml = await (await fetch(src)).text();
      /* offline bundles swap remote icon urls for inlined copies (as percent-encoded
         data uris: drawio style strings are ';'-delimited, so no base64 marker) */
      if (window.__resources && window.__xmlRewrites) {
        for (let i = 0; i < window.__xmlRewrites.length; i++) {
          const pair = window.__xmlRewrites[i];
          const res = window.__resources[pair[1]];
          if (!res) continue;
          let replacement = res;
          if (/^blob:/.test(res)) {
            try {
              const text = await (await fetch(res)).text();
              replacement = 'data:image/svg+xml,' + encodeURIComponent(text);
            } catch (e) { continue; }
          }
          xml = xml.split(pair[0]).join(replacement);
        }
      }
      const mount = document.createElement('div');
      mount.className = 'mxgraph';
      mount.style.cssText = 'width:100%;height:100%';
      mount.setAttribute('data-mxgraph', JSON.stringify({
        highlight: '#ffffff', nav: false, resize: true, toolbar: null,
        'dark-mode': 'light', border: 24, center: true, xml: xml
      }));
      this.host.appendChild(mount);
      const self = this;
      window.GraphViewer.createViewerForElement(mount, function (viewer) {
        self.viewer = viewer;
        self.graph = viewer.graph;
        self.collect();
        self.reset();
        self.refit();
        self.host.style.opacity = '1';
        self.observe();
        self.dispatchEvent(new CustomEvent('ds-ready', { detail: { total: self.steps.length } }));
        if (String(self.attr('autoplay')) !== 'false') self.play();
      });
    }

    /* every cell touched by the timeline, groups expanded */
    collect() {
      const model = this.graph.model;
      const expand = (id) => {
        const cell = model.getCell(id);
        if (!cell) { console.warn('[diagram-stage] unknown cell', id); return []; }
        let out = [id];
        const n = model.getChildCount(cell);
        for (let i = 0; i < n; i++) out = out.concat(expand(model.getChildAt(cell, i).id));
        return out;
      };
      this.stepCells = this.steps.map(function (s) {
        return s.ids.map(expand);
      });
      this.allCells = [];
      this.stepCells.forEach((groups) => groups.forEach((ids) => { this.allCells = this.allCells.concat(ids); }));
    }

    nodesFor(id) {
      const cell = this.graph.model.getCell(id);
      if (!cell) return [];
      const st = this.graph.view.getState(cell);
      if (!st) return [];
      const out = [];
      if (st.shape && st.shape.node) out.push(st.shape.node);
      if (st.text && st.text.node) out.push(st.text.node);
      return out;
    }

    setVisible(id, visible, delaySec) {
      const fade = this.num('fade', 0.55);
      this.nodesFor(id).forEach(function (node) {
        if (node.__dsOrig === undefined) node.__dsOrig = node.style.opacity || '';
        node.style.transition = visible
          ? 'opacity ' + fade + 's ease-out ' + (delaySec || 0) + 's'
          : 'none';
        node.style.opacity = visible ? (node.__dsOrig || '1') : '0';
      });
    }

    applyState() {
      if (!this.graph) return;
      const self = this;
      this.allCells.forEach(function (id) { self.setVisible(id, self.revealed.has(id), 0); });
    }

    /* the diagram is much wider than 16:9, so the fit leaves a band below it */
    /* a slide that mounted hidden fit the graph to a zero box, and the viewer's own
       fit refuses to scale a small diagram up — so fit by hand */
    /* fit against the RENDERED svg ink: cell geometry bounds under-report
       labels and overlays, which left the diagram bleeding past its box */
    refit() {
      if (!this.graph) return;
      const W = this.clientWidth, H = this.clientHeight;
      if (W < 40 || H < 40) return;
      if (this.viewer && typeof this.viewer.fitGraph === 'function') {
        try { this.viewer.fitGraph(); } catch (e) { /* keep current fit */ }
      }
      const view = this.graph.view;
      const b = this.graph.getGraphBounds();
      if (b && b.width > 1 && b.height > 1) {
        const pad = 28;
        const f = Math.min((W - pad * 2) / b.width, (H - pad * 2) / b.height);
        if (f > 0 && Math.abs(f - 1) > 0.02) view.setScale(view.scale * f);
      }
      this.recenter();
      this.applyState();
    }

    recenter() {
      if (!this.graph) return;
      const b = this.graph.getGraphBounds();
      if (!b || !b.height) return;
      const dx = (this.clientWidth - b.width) / 2 - b.x;
      const dy = (this.clientHeight - b.height) / 2 - b.y;
      this.host.style.transform = 'translate(' + Math.round(dx) + 'px,' + Math.round(dy) + 'px)';
    }

    observe() {
      const self = this;
      let t = null;
      const ro = new ResizeObserver(function () {
        clearTimeout(t);
        t = setTimeout(function () { self.refit(); }, 120);
      });
      ro.observe(this);
    }

    reset() {
      clearTimeout(this.timer);
      this.timer = null;
      this.index = 0;
      this.revealed = new Set();
      this.applyState();
      this.emitStep();
    }

    emitStep() {
      this.dispatchEvent(new CustomEvent('ds-step', {
        detail: { index: this.index, total: this.steps.length, playing: !!this.timer }
      }));
    }

    revealStep(i) {
      const step = this.steps[i];
      if (!step) return;
      const stagger = step.stagger || 0;
      const self = this;
      this.stepCells[i].forEach(function (ids, k) {
        ids.forEach(function (id) {
          self.revealed.add(id);
          self.setVisible(id, true, k * stagger);
        });
      });
    }

    gapAfter(i) {
      const step = this.steps[i];
      const stagger = (step.stagger || 0) * Math.max(0, step.ids.length - 1);
      const kind = step.after || 'step';
      const base = kind === 'region' ? this.num('region-gap', 0.5)
        : kind === 'block' ? this.num('block-gap', 1)
          : this.num('step', 1);
      return (stagger + base) * 1000;
    }

    tick() {
      if (this.index >= this.steps.length) { this.finish(); return; }
      const i = this.index;
      this.revealStep(i);
      this.index = i + 1;
      this.emitStep();
      const self = this;
      this.timer = setTimeout(function () { self.tick(); }, this.gapAfter(i));
    }

    finish() {
      clearTimeout(this.timer);
      this.timer = null;
      this.emitStep();
      this.dispatchEvent(new CustomEvent('ds-end'));
    }

    play() {
      if (!this.graph) return;
      this.refit();
      this.reset();
      this.tick();
    }

    pause() {
      clearTimeout(this.timer);
      this.timer = null;
      this.emitStep();
    }

    resume() {
      if (this.timer || !this.graph) return;
      if (this.index >= this.steps.length) return;
      this.tick();
    }

    toggle() { this.timer ? this.pause() : this.resume(); }

    showAll() {
      clearTimeout(this.timer);
      this.timer = null;
      const self = this;
      this.allCells.forEach(function (id) { self.revealed.add(id); });
      this.index = this.steps.length;
      this.applyState();
      this.emitStep();
    }
  }

  if (!window.customElements.get('diagram-stage')) {
    window.customElements.define('diagram-stage', DiagramStage);
  }
})();
