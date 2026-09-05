/* ═══════════════════════════════════════════════════════════════════
   SO-101 Dashboard — Client Logic
   ═══════════════════════════════════════════════════════════════════ */

// ─── State ────────────────────────────────────────────────────────
let isConnected = false;
let isConnecting = false;
let reachyOnline = false;
let tableZ = parseFloat(localStorage.getItem('tableZ') || '0.02'); // default 2cm, calibratable

// ─── Tab Switching ────────────────────────────────────────────────
let activeTab = 'logi';

function switchTab(tabName) {
  activeTab = tabName;
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  const tabBtn = document.querySelector(`.tab[data-tab="${tabName}"]`);
  if (tabBtn) tabBtn.classList.add('active');

  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  const panel = document.getElementById(`tab-${tabName}`);
  if (panel) panel.classList.add('active');

  if (tabName === 'logi') {
    setTimeout(initLogiGridCanvas, 50);
  } else if (tabName === 'ai') {
    switchAISubTab(aiCurrentSubTab);
  }
}

// ─── API Helper ───────────────────────────────────────────────────
async function api(endpoint, method = 'GET', body = null) {
  const opts = { method, headers: {} };
  if (body) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  try {
    const res = await fetch(`/api/${endpoint}`, opts);
    return await res.json();
  } catch (err) {
    console.error(`API error [${endpoint}]:`, err);
    return { status: 'error', message: 'Server unreachable' };
  }
}

// ─── Toast Notifications ──────────────────────────────────────────
function toast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = message;
  container.appendChild(el);
  setTimeout(() => {
    if (el.parentNode) el.remove();
  }, 3200);
}

// ─── Connection ───────────────────────────────────────────────────
async function toggleConnection(mode = 'follower_only') {
  const btnFollower = document.getElementById('btn-connect-follower');
  const btnBoth = document.getElementById('btn-connect');
  const pill = document.getElementById('status-pill');
  const statusText = document.getElementById('status-text');

  if (isConnected) {
    // Disconnect
    if (btnFollower) btnFollower.disabled = true;
    if (btnBoth) btnBoth.disabled = true;
    try {
      const res = await api('disconnect', 'POST');
      isConnected = false;
      toast('Arms disconnected', 'info');
      if (pill) pill.className = 'status-pill disconnected';
      if (statusText) statusText.textContent = 'Arms: Offline';
      if (btnFollower) { btnFollower.textContent = '🦾 Connect Follower Only'; btnFollower.className = 'btn btn-primary'; }
      if (btnBoth) { btnBoth.textContent = '🔗 Connect Both Arms'; btnBoth.className = 'btn btn-outline'; }
    } catch (err) {
      toast('Disconnect request failed', 'error');
    } finally {
      if (btnFollower) btnFollower.disabled = false;
      if (btnBoth) btnBoth.disabled = false;
    }
  } else {
    // Connect
    isConnecting = true;
    const targetBtn = (mode === 'follower_only') ? btnFollower : btnBoth;
    if (targetBtn) {
      targetBtn.disabled = true;
      targetBtn.textContent = 'Connecting...';
    }

    try {
      const res = await api('connect', 'POST', { mode });
      isConnecting = false;

      if (res.status === 'connected' || res.status === 'already_connected') {
        isConnected = true;
        const modeLabel = (mode === 'follower_only') ? 'Follower Online' : 'Both Arms Online';
        toast(`Connected successfully (${modeLabel})!`, 'success');
        if (pill) pill.className = 'status-pill connected';
        if (statusText) statusText.textContent = `Arms: ${modeLabel}`;
        if (btnFollower) btnFollower.textContent = 'Disconnect';
        if (btnBoth) btnBoth.textContent = 'Disconnect';
      } else {
        toast(res.message || 'Connection failed', 'error');
        if (pill) pill.className = 'status-pill disconnected';
        if (statusText) statusText.textContent = 'Arms: Offline';
        btn.textContent = 'Connect Arms';
      }
    } catch (err) {
      isConnecting = false;
      toast('Connection request failed', 'error');
      if (pill) pill.className = 'status-pill disconnected';
      if (statusText) statusText.textContent = 'Arms: Offline';
      btn.textContent = 'Connect Arms';
    } finally {
      btn.disabled = false;
    }
  }
  await pollStatus();
}
// ─── Teleoperation ────────────────────────────────────────────────
async function startTeleop() {
  const res = await api('start_teleop', 'POST');
  if (res.status === 'teleop_started') {
    toast('Teleop started — move the leader arm', 'success');
  } else {
    toast(res.message || 'Cannot start teleop', 'error');
  }
  await pollStatus();
}

async function stopTeleop() {
  await api('stop_teleop', 'POST');
  toast('Stopping teleop...', 'info');
  await pollStatus();
}

// ─── Recording ────────────────────────────────────────────────────
async function startRecording() {
  const res = await api('start_recording', 'POST');
  if (res.status === 'recording') {
    toast('Recording started — move the leader arm', 'success');
  } else {
    toast(res.message || 'Cannot start recording', 'error');
  }
  await pollStatus();
}

async function stopRecording() {
  const res = await api('stop_recording', 'POST');
  if (res.step_count !== undefined) {
    toast(`Recording stopped — ${res.step_count} steps captured`, 'info');
  } else {
    toast('Recording stopped', 'info');
  }
  await pollStatus();
}

// ─── Playback ─────────────────────────────────────────────────────
async function playRecording() {
  const res = await api('play_recording', 'POST');
  if (res.status === 'playing') {
    toast('Playback started', 'success');
  } else {
    toast(res.message || 'Cannot play recording', 'error');
  }
  await pollStatus();
}

async function stopPlayback() {
  await api('stop_playback', 'POST');
  toast('Stopping playback...', 'info');
  await pollStatus();
}

// ─── Move to XYZ ──────────────────────────────────────────────────
async function moveToXYZ() {
  const x = parseFloat(document.getElementById('input-x').value);
  const y = parseFloat(document.getElementById('input-y').value);
  const z = parseFloat(document.getElementById('input-z').value);
  const duration = parseFloat(document.getElementById('input-speed').value);

  if (isNaN(x) || isNaN(y) || isNaN(z)) {
    toast('Please enter valid coordinates', 'error');
    return;
  }

  const statusBox = document.getElementById('move-status');
  statusBox.textContent = `Moving to (${x.toFixed(2)}, ${y.toFixed(2)}, ${z.toFixed(2)}) in ${duration}s...`;
  statusBox.classList.add('active');

  const res = await api('move_xyz', 'POST', { x, y, z, duration });
  if (res.status === 'moving') {
    toast(`Moving to (${x}, ${y}, ${z}) in ${duration}s`, 'success');
  } else {
    toast(res.message || 'Cannot move', 'error');
    statusBox.textContent = 'Move failed';
    statusBox.classList.remove('active');
  }
  await pollStatus();
}

function updateSpeedLabel(val) {
  document.getElementById('speed-label').textContent = `${parseFloat(val).toFixed(1)}s`;
}

async function stopMove() {
  await api('stop_move', 'POST');
  toast('Stopping movement...', 'info');
  document.getElementById('move-status').textContent = 'Stopping...';
  await pollStatus();
}

// ─── Home ─────────────────────────────────────────────────────────
async function goHome() {
  const res = await api('home', 'POST');
  if (res.status === 'moving_home') {
    toast('Returning to home position...', 'info');
  } else {
    toast(res.message || 'Cannot go home', 'error');
  }
  await pollStatus();
}

// ─── Status Polling ───────────────────────────────────────────────
async function pollStatus() {
  try {
    const s = await api('status');
    if (s.status !== 'error') {
      updateUI(s);
    }
  } catch (e) {
    // Server might be down
  }
}

function updateUI(s) {
  isConnected = s.connected;
  reachyOnline = s.reachy_online;

  // ── Connection status ──
  const pill = document.getElementById('status-pill');
  const statusText = document.getElementById('status-text');
  const btnFollower = document.getElementById('btn-connect-follower');
  const btnBoth = document.getElementById('btn-connect');

  if (!isConnecting) {
    pill.className = `status-pill ${s.connected ? 'connected' : 'disconnected'}`;
    const modeLabel = s.connection_mode === 'follower_only' ? 'Follower Online' : (s.connected ? 'Both Online' : 'Arms: Offline');
    if (statusText) statusText.textContent = s.connected ? `Arms: ${modeLabel}` : 'Arms: Offline';
    if (btnFollower) btnFollower.textContent = s.connected ? 'Disconnect' : '🦾 Connect Follower Only';
    if (btnBoth) btnBoth.textContent = s.connected ? 'Disconnect' : '🔗 Connect Both Arms';
  }

  // ── Reachy connection status ──
  const reachyPill = document.getElementById('reachy-status-pill');
  const warningBanner = document.getElementById('reachy-offline-warning');
  
  if (reachyPill) {
    reachyPill.className = `status-pill ${s.reachy_online ? 'connected' : 'disconnected'}`;
    const dot = reachyPill.querySelector('.status-dot');
    if (dot) {
      dot.style.backgroundColor = s.reachy_online ? '#10b981' : 'var(--text-muted)';
    }
    reachyPill.querySelector('span:not(.status-dot)').textContent = s.reachy_online ? 'Reachy: Online' : 'Reachy: Offline';
  }

  if (warningBanner) {
    if (s.reachy_online) {
      warningBanner.classList.add('hidden');
    } else {
      warningBanner.classList.remove('hidden');
    }
  }

  // Update Reachy Camera Tab panels
  const placeholder = document.getElementById('reachy-cam-placeholder');
  const streamContainer = document.getElementById('reachy-cam-stream-container');
  const connBtn = document.getElementById('btn-connect-reachy');
  const forceBtn = document.getElementById('btn-force-reachy');
  const discBtn = document.getElementById('btn-disconnect-reachy');
  const ipInput = document.getElementById('input-reachy-ip');

  if (s.reachy_online) {
    if (placeholder) placeholder.classList.add('hidden');
    if (streamContainer) streamContainer.classList.remove('hidden');
    if (connBtn) connBtn.classList.add('hidden');
    if (forceBtn) forceBtn.classList.add('hidden');
    if (discBtn) discBtn.classList.remove('hidden');
    if (ipInput) ipInput.disabled = true;
    
    // Set video stream image src
    const streamImg = document.getElementById('reachy-stream-img');
    if (streamImg && s.reachy_ip) {
      // Stream via local proxy that converts gRPC frames to HTTP MJPEG
      const targetSrc = '/api/reachy_camera';
      if (!streamImg.src.includes(targetSrc) || streamImg.getAttribute('data-error') === '1') {
        streamImg.removeAttribute('data-error');
        streamImg.src = targetSrc + '?t=' + Date.now();
      }
      streamImg.style.display = 'block';
    }
    
    // Update displayed IP & camera FPS
    if (s.camera_fps) {
      window.latestCameraFps = s.camera_fps;
    }
    document.querySelectorAll('.reachy-ip-display').forEach(el => {
      el.textContent = s.reachy_ip;
    });
  } else {
    if (placeholder) placeholder.classList.remove('hidden');
    if (streamContainer) streamContainer.classList.add('hidden');
    if (connBtn) connBtn.classList.remove('hidden');
    if (forceBtn) forceBtn.classList.remove('hidden');
    if (discBtn) discBtn.classList.add('hidden');
    if (ipInput) ipInput.disabled = false;
    
    // Clear image stream
    const streamImg = document.getElementById('reachy-stream-img');
    if (streamImg) streamImg.src = '';
  }

  // ── Teleop indicator ──
  const teleopIndicator = document.getElementById('teleop-indicator');
  const teleopCard = document.querySelector('.teleop-card');
  if (s.teleop) {
    teleopIndicator.classList.remove('hidden');
    teleopCard.classList.add('active');
  } else {
    teleopIndicator.classList.add('hidden');
    teleopCard.classList.remove('active');
  }

  // ── Teleop buttons ──
  const canStartTeleop = s.connected && !s.teleop && !s.playing && !s.moving;
  document.getElementById('btn-start-teleop').disabled = !canStartTeleop;
  document.getElementById('btn-stop-teleop').disabled = !s.teleop;

  // ── Step count ──
  document.getElementById('step-count').textContent = s.step_count;

  // ── Recording indicator ──
  const recIndicator = document.getElementById('recording-indicator');
  if (s.recording) {
    recIndicator.classList.remove('hidden');
  } else {
    recIndicator.classList.add('hidden');
  }

  // ── Recording buttons ──
  const canRecord = s.connected && !s.recording && !s.playing && !s.moving;
  document.getElementById('btn-start-rec').disabled = !canRecord;
  document.getElementById('btn-stop-rec').disabled = !s.recording;

  // ── Playback buttons ──
  const idle = s.connected && !s.teleop && !s.recording && !s.playing && !s.moving;
  document.getElementById('btn-play').disabled = !idle || s.step_count === 0;
  document.getElementById('btn-stop-play').disabled = !s.playing;
  document.getElementById('btn-home-rec').disabled = !idle;

  // ── Move buttons ──
  document.getElementById('btn-move').disabled = !idle;
  document.getElementById('btn-stop-move').disabled = !s.moving;
  document.getElementById('btn-home-move').disabled = !idle;

  // ── Move status box ──
  const statusBox = document.getElementById('move-status');
  if (!s.moving && statusBox.classList.contains('active')) {
    statusBox.textContent = 'Move complete ✓';
    statusBox.classList.remove('active');
  } else if (!s.moving && !statusBox.classList.contains('active') && s.connected) {
    if (statusBox.textContent.includes('connect')) {
      statusBox.textContent = 'Ready';
    }
  }

  // ── Joint angles ──
  renderAngles('leader-angles', s.leader_angles, s.connected);
  renderAngles('follower-angles', s.follower_angles, s.connected);

  // ── Downloads status ──
  if (s.downloads) {
    updateDownloadUI('yolov8', s.downloads.yolov8);
  }
}

function updateDownloadUI(id, dl) {
  const btn = document.getElementById(`btn-dl-${id}`);
  const progressContainer = document.getElementById(`progress-${id}`);
  const bar = document.getElementById(`bar-${id}`);
  const speed = document.getElementById(`speed-${id}`);
  const percent = document.getElementById(`percent-${id}`);

  if (!btn || !progressContainer) return;

  if (dl.status === 'idle') {
    btn.classList.remove('hidden');
    btn.disabled = !reachyOnline;
    btn.textContent = 'Deploy to Reachy';
    progressContainer.classList.add('hidden');
  } else if (dl.status === 'downloading') {
    btn.classList.add('hidden');
    progressContainer.classList.remove('hidden');
    bar.style.width = `${dl.progress}%`;
    percent.textContent = `${dl.progress}%`;
    speed.textContent = dl.speed;
  } else if (dl.status === 'completed') {
    btn.classList.remove('hidden');
    btn.disabled = true;
    btn.textContent = 'Deployed ✓';
    btn.className = 'btn btn-secondary full-width-btn';
    progressContainer.classList.add('hidden');
  } else if (dl.status === 'error') {
    btn.classList.remove('hidden');
    btn.disabled = !reachyOnline;
    btn.textContent = 'Retry Deployment';
    btn.className = 'btn btn-accent full-width-btn';
    progressContainer.classList.add('hidden');
  }
}

async function downloadModel(modelId) {
  toast(`Starting deployment for ${modelId}...`, 'info');
  const res = await api('download_model', 'POST', { model: modelId });
  if (res.status === 'started') {
    toast(`Transferring ${modelId} to Reachy in background`, 'success');
  } else if (res.status === 'already_downloading') {
    toast(`Already transferring ${modelId}`, 'info');
  } else {
    toast(res.message || `Failed to start transfer`, 'error');
  }
  await pollStatus();
}

async function connectReachy(force = false) {
  const ipInput = document.getElementById('input-reachy-ip');
  if (!ipInput) return;
  const ip = ipInput.value.trim();
  if (!ip) {
    toast('Please enter a Reachy IP address or hostname', 'error');
    return;
  }

  toast(force ? 'Force connecting...' : `Connecting to ${ip}...`, 'info');
  
  const connBtn = document.getElementById('btn-connect-reachy');
  const forceBtn = document.getElementById('btn-force-reachy');
  if (connBtn) connBtn.disabled = true;
  if (forceBtn) forceBtn.disabled = true;

  try {
    const res = await api('connect_reachy', 'POST', { ip, force });
    if (res.status === 'connected') {
      toast(`Successfully connected to Reachy at ${res.ip}!`, 'success');
    } else {
      toast(res.message || 'Could not connect to Reachy', 'error');
    }
  } catch (err) {
    toast('Connection failed', 'error');
  } finally {
    if (connBtn) connBtn.disabled = false;
    if (forceBtn) forceBtn.disabled = false;
    await pollStatus();
  }
}

async function disconnectReachy() {
  toast('Disconnecting Reachy...', 'info');
  try {
    await api('disconnect_reachy', 'POST');
    toast('Disconnected Reachy Mini', 'success');
  } catch (err) {
    toast('Disconnect failed', 'error');
  } finally {
    await pollStatus();
  }
}

function renderAngles(elementId, angles, connected) {
  const container = document.getElementById(elementId);

  if (!connected) {
    container.innerHTML = '<span class="no-data">Connect to view angles</span>';
    return;
  }

  if (!angles || Object.keys(angles).length === 0) {
    container.innerHTML = '<span class="no-data">No data yet</span>';
    return;
  }

  container.innerHTML = Object.entries(angles)
    .map(([name, val]) => {
      const cleanName = name.replace('.pos', '');
      return `<div class="angle-item">
        <span class="angle-name">${cleanName}</span>
        <span class="angle-value">${Number(val).toFixed(1)}°</span>
      </div>`;
    })
    .join('');
}

// ─── Reachy Control Panel (RCP) ────────────────────────────────────
// Converts degrees to radians for display
function deg2rad(d) { return d * Math.PI / 180; }

// Update slider fill gradient
function updateSliderFill(slider) {
  const min = parseFloat(slider.min);
  const max = parseFloat(slider.max);
  const val = parseFloat(slider.value);
  const pct = ((val - min) / (max - min)) * 100;
  slider.style.background = `linear-gradient(to right, #f90 ${pct}%, #555 ${pct}%)`;
}

// Draw arc gauge on a canvas
function drawArcGauge(canvasId, valueDeg, maxDeg) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  const cx = w / 2, cy = h / 2, r = Math.min(w, h) / 2 - 4;
  
  ctx.clearRect(0, 0, w, h);
  
  // Background arc
  ctx.beginPath();
  ctx.arc(cx, cy, r, Math.PI * 0.75, Math.PI * 2.25);
  ctx.strokeStyle = '#444';
  ctx.lineWidth = 4;
  ctx.lineCap = 'round';
  ctx.stroke();
  
  // Value arc
  const totalArc = Math.PI * 1.5;
  const norm = (valueDeg + maxDeg) / (2 * maxDeg); // 0..1
  const endAngle = Math.PI * 0.75 + totalArc * Math.max(0, Math.min(1, norm));
  
  ctx.beginPath();
  ctx.arc(cx, cy, r, Math.PI * 0.75, endAngle);
  ctx.strokeStyle = '#f90';
  ctx.lineWidth = 4;
  ctx.lineCap = 'round';
  ctx.stroke();
}

// ─── Antenna Controls ──────────────────────────────────────────────
function setupAntennaSlider(side) {
  const slider = document.getElementById(`rcp-antenna-${side}`);
  const valEl = document.getElementById(`rcp-antenna-${side}-val`);
  const arcId = `rcp-arc-antenna-${side}`;
  if (!slider) return;
  
  const update = () => {
    const deg = parseFloat(slider.value);
    const rad = deg2rad(deg);
    valEl.textContent = `${rad.toFixed(3)} rad`;
    updateSliderFill(slider);
    drawArcGauge(arcId, deg, 180);
  };
  
  slider.addEventListener('input', update);
  slider.addEventListener('change', () => {
    update();
    sendAntennaMove();
  });
  
  update();
}

async function sendAntennaMove() {
  const leftDeg = parseFloat(document.getElementById('rcp-antenna-left')?.value || 0);
  const rightDeg = parseFloat(document.getElementById('rcp-antenna-right')?.value || 0);
  
  try {
    await api('move_antennas', 'POST', { left_angle: leftDeg, right_angle: rightDeg });
  } catch (err) {
    // suppress if offline
  }
}

// ─── Head Joystick Pad ─────────────────────────────────────────────
let headPadDragging = false;
let headPadX = 0; // -1..1 range
let headPadY = 0;

function initHeadPad() {
  const canvas = document.getElementById('rcp-pad-head');
  if (!canvas) return;
  
  drawHeadPad();
  
  canvas.addEventListener('pointerdown', (e) => {
    headPadDragging = true;
    canvas.setPointerCapture(e.pointerId);
    updateHeadPadFromEvent(e);
  });
  
  canvas.addEventListener('pointermove', (e) => {
    if (!headPadDragging) return;
    updateHeadPadFromEvent(e);
  });
  
  canvas.addEventListener('pointerup', (e) => {
    headPadDragging = false;
    canvas.releasePointerCapture(e.pointerId);
    sendHeadMoveFromPad();
  });
}

let lastHeadMoveTime = 0;
function sendHeadMoveFromPadThrottled() {
  const now = Date.now();
  if (now - lastHeadMoveTime > 60) {
    lastHeadMoveTime = now;
    sendHeadMoveFromPad();
  }
}

function updateHeadPadFromEvent(e) {
  const canvas = document.getElementById('rcp-pad-head');
  const rect = canvas.getBoundingClientRect();
  const x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  const y = ((e.clientY - rect.top) / rect.height) * 2 - 1;
  
  headPadX = Math.max(-1, Math.min(1, x));
  headPadY = Math.max(-1, Math.min(1, y));
  
  drawHeadPad();
  updateHeadPadDisplay();
  sendHeadMoveFromPadThrottled();
}

function drawHeadPad() {
  const canvas = document.getElementById('rcp-pad-head');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  const cx = w / 2, cy = h / 2;
  const r = Math.min(w, h) / 2 - 12;
  
  ctx.clearRect(0, 0, w, h);
  
  // Outer circle (dotted)
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.strokeStyle = 'rgba(255, 153, 0, 0.35)';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 4]);
  ctx.stroke();
  ctx.setLineDash([]);
  
  // Crosshairs
  ctx.strokeStyle = 'rgba(255, 153, 0, 0.25)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(cx - r, cy);
  ctx.lineTo(cx + r, cy);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(cx, cy - r);
  ctx.lineTo(cx, cy + r);
  ctx.stroke();
  
  // Thumb position
  const tx = cx + headPadX * r;
  const ty = cy + headPadY * r;
  
  // Line from center to thumb
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(tx, ty);
  ctx.strokeStyle = 'rgba(255, 153, 0, 0.5)';
  ctx.lineWidth = 2;
  ctx.stroke();
  
  // Thumb dot
  ctx.beginPath();
  ctx.arc(tx, ty, 8, 0, Math.PI * 2);
  ctx.fillStyle = '#f90';
  ctx.fill();
  ctx.beginPath();
  ctx.arc(tx, ty, 8, 0, Math.PI * 2);
  ctx.strokeStyle = 'rgba(255, 153, 0, 0.6)';
  ctx.lineWidth = 2;
  ctx.stroke();
}

function updateHeadPadDisplay() {
  const valEl = document.getElementById('rcp-head-xy-val');
  if (valEl) {
    const yawDeg = headPadX * 60;
    const pitchDeg = headPadY * -30; // Inverted: up = positive pitch
    valEl.textContent = `Yaw: ${yawDeg.toFixed(1)}°  Pitch: ${pitchDeg.toFixed(1)}°`;
  }
}

async function sendHeadMoveFromPad() {
  const yaw = headPadX * 60;   // -60..60 deg
  const pitch = headPadY * -30; // -30..30 deg (inverted)
  const roll = parseFloat(document.getElementById('rcp-head-roll')?.value || 0);
  const z = parseFloat(document.getElementById('rcp-head-z')?.value || 0);

  try {
    await api('move_head', 'POST', { yaw, pitch, roll, z, duration: 0.1 });
  } catch (err) {
    // suppress if offline
  }
}

function setupHeadZAndRollSliders() {
  const zSlider = document.getElementById('rcp-head-z');
  const zVal = document.getElementById('rcp-head-z-val');
  const rollSlider = document.getElementById('rcp-head-roll');
  const rollVal = document.getElementById('rcp-head-roll-val');

  if (zSlider && zVal) {
    const updateZ = () => {
      const val = parseFloat(zSlider.value);
      const meters = val / 1000.0;
      zVal.textContent = `${meters.toFixed(3)} m`;
      updateSliderFill(zSlider);
    };
    zSlider.addEventListener('input', () => {
      updateZ();
      sendHeadMoveFromPadThrottled();
    });
    zSlider.addEventListener('change', () => {
      updateZ();
      sendHeadMoveFromPad();
    });
    updateZ();
  }

  if (rollSlider && rollVal) {
    const updateRoll = () => {
      const deg = parseFloat(rollSlider.value);
      const rad = deg2rad(deg);
      rollVal.textContent = `${rad.toFixed(3)} rad`;
      updateSliderFill(rollSlider);
    };
    rollSlider.addEventListener('input', () => {
      updateRoll();
      sendHeadMoveFromPadThrottled();
    });
    rollSlider.addEventListener('change', () => {
      updateRoll();
      sendHeadMoveFromPad();
    });
    updateRoll();
  }
}

// ─── Torso Control ─────────────────────────────────────────────────
function setupTorsoSlider() {
  const slider = document.getElementById('rcp-torso-yaw');
  const valEl = document.getElementById('rcp-torso-yaw-val');
  const arcId = 'rcp-arc-torso';
  if (!slider) return;
  
  const update = () => {
    const deg = parseFloat(slider.value);
    const rad = deg2rad(deg);
    valEl.textContent = `${rad.toFixed(3)} rad`;
    updateSliderFill(slider);
    drawArcGauge(arcId, deg, 45);
  };
  
  slider.addEventListener('input', update);
  slider.addEventListener('change', () => {
    update();
    sendTorsoMove();
  });
  
  update();
}

async function sendTorsoMove() {
  const yawDeg = parseFloat(document.getElementById('rcp-torso-yaw')?.value || 0);
  
  try {
    await api('move_torso', 'POST', { yaw: yawDeg });
  } catch (err) {
    // suppress if offline
  }
}

// ─── Reset All ─────────────────────────────────────────────────────
async function rcpResetAll() {
  // Reset UI controls
  ['left', 'right'].forEach(side => {
    const slider = document.getElementById(`rcp-antenna-${side}`);
    if (slider) {
      slider.value = 0;
      slider.dispatchEvent(new Event('input'));
    }
  });
  
  headPadX = 0;
  headPadY = 0;
  drawHeadPad();
  updateHeadPadDisplay();

  const zSlider = document.getElementById('rcp-head-z');
  if (zSlider) {
    zSlider.value = 0;
    zSlider.dispatchEvent(new Event('input'));
  }

  const rollSlider = document.getElementById('rcp-head-roll');
  if (rollSlider) {
    rollSlider.value = 0;
    rollSlider.dispatchEvent(new Event('input'));
  }
  
  const torso = document.getElementById('rcp-torso-yaw');
  if (torso) {
    torso.value = 0;
    torso.dispatchEvent(new Event('input'));
  }
  
  // Send ONE combined API call instead of separate ones
  try {
    await api('reset_all', 'POST', { duration: 1.0 });
    toast('All controls reset to center', 'info');
  } catch (err) {
    // suppress if offline
  }
}

// ─── Initialize RCP on Load ────────────────────────────────────────
function initRCP() {
  setupAntennaSlider('left');
  setupAntennaSlider('right');
  initHeadPad();
  setupHeadZAndRollSliders();
  setupTorsoSlider();
}

// Backward compatibility stubs for old functions
function updateHeadVal() {}
function sendHeadMove() { sendHeadMoveFromPad(); }
function nudgeHead(yawOff, pitchOff) {
  headPadX = Math.max(-1, Math.min(1, headPadX + yawOff / 60));
  headPadY = Math.max(-1, Math.min(1, headPadY - pitchOff / 30));
  drawHeadPad();
  updateHeadPadDisplay();
  sendHeadMoveFromPad();
}
function resetHead() {
  headPadX = 0;
  headPadY = 0;
  drawHeadPad();
  updateHeadPadDisplay();
  sendHeadMoveFromPad();
}

// ─── Init ─────────────────────────────────────────────────────────
pollStatus();
setInterval(pollStatus, 1000);

// Initialize Reachy Control Panel
initRCP();

// ─── Live FPS Counter ─────────────────────────────────────────────
(function initFpsCounter() {
  function updateFpsDisplay() {
    const fpsEl = document.getElementById('fps-counter');
    if (fpsEl) {
      if (window.latestCameraFps) {
        fpsEl.textContent = `${window.latestCameraFps} FPS`;
      } else {
        fpsEl.textContent = `30.0 FPS`;
      }
    }
  }
  setInterval(updateFpsDisplay, 400);
})();

// ─── Voice Assistant Management JS ────────────────────────────────
let voiceLastLogId = 0;
let voiceIsRunning = false;
let voicePollInterval = null;

async function startVoiceAssistant() {
  const btnStart = document.getElementById('btn-voice-start');
  const btnStop = document.getElementById('btn-voice-stop');
  if (btnStart) btnStart.disabled = true;

  try {
    const res = await api('voice/start', 'POST');
    if (res.status === 'success') {
      toast('Voice Assistant started!', 'success');
      updateVoiceStatusUI(true);
    } else {
      toast(res.message || 'Failed to start Voice Assistant', 'error');
      if (btnStart) btnStart.disabled = false;
    }
  } catch (err) {
    toast('Error starting Voice Assistant', 'error');
    if (btnStart) btnStart.disabled = false;
  }
}

async function stopVoiceAssistant() {
  const btnStop = document.getElementById('btn-voice-stop');
  if (btnStop) btnStop.disabled = true;

  try {
    const res = await api('voice/stop', 'POST');
    if (res.status === 'success') {
      toast('Voice Assistant stopped.', 'info');
      updateVoiceStatusUI(false);
    } else {
      toast(res.message || 'Failed to stop Voice Assistant', 'error');
      if (btnStop) btnStop.disabled = false;
    }
  } catch (err) {
    toast('Error stopping Voice Assistant', 'error');
    if (btnStop) btnStop.disabled = false;
  }
}

function clearVoiceLogs() {
  const terminalContent = document.getElementById('voice-terminal-content');
  if (terminalContent) {
    terminalContent.innerHTML = '<span class="vt-system">[System] Logs cleared.</span>';
  }
}

function updateVoiceStatusUI(running) {
  voiceIsRunning = running;
  const badge = document.getElementById('voice-status-badge');
  const badgeText = document.getElementById('voice-status-text');
  const btnStart = document.getElementById('btn-voice-start');
  const btnStop = document.getElementById('btn-voice-stop');

  if (badge && badgeText) {
    if (running) {
      badge.className = 'voice-badge running';
      badgeText.textContent = 'Running';
    } else {
      badge.className = 'voice-badge offline';
      badgeText.textContent = 'Offline';
    }
  }

  if (btnStart) btnStart.disabled = running;
  if (btnStop) btnStop.disabled = !running;
}

function colorizeVoiceLine(text) {
  let cssClass = 'vt-system';
  if (text.includes('[You]')) cssClass = 'vt-you';
  else if (text.includes('[Gemini]')) cssClass = 'vt-gemini';
  else if (text.includes('[Function Call]')) cssClass = 'vt-function';
  else if (text.includes('[Function Result]')) cssClass = 'vt-result';
  else if (text.includes('[Reachy Mini]')) cssClass = 'vt-robot';
  else if (text.includes('[Voice] Listening')) cssClass = 'vt-listening';
  else if (text.includes('[Voice]')) cssClass = 'vt-voice';
  else if (text.includes('[Clock Tool]')) cssClass = 'vt-clock';
  else if (text.includes('[Weather Tool]')) cssClass = 'vt-weather';
  else if (text.includes('[Timer Tool]')) cssClass = 'vt-timer';
  else if (text.toLowerCase().includes('error') || text.toLowerCase().includes('exception')) cssClass = 'vt-error';

  const div = document.createElement('div');
  div.className = `vt-line ${cssClass}`;
  div.textContent = text;
  return div;
}

async function pollVoiceStatusAndLogs() {
  try {
    // Poll status
    const statusRes = await api('voice/status', 'GET');
    if (statusRes && typeof statusRes.running === 'boolean') {
      updateVoiceStatusUI(statusRes.running);
    }

    // Poll logs
    const logsRes = await api(`voice/logs?since=${voiceLastLogId}`, 'GET');
    if (logsRes && logsRes.lines && logsRes.lines.length > 0) {
      const container = document.getElementById('voice-terminal-content');
      if (container) {
        // If initial placeholder is present, clear it
        if (container.querySelector('.vt-system') && container.children.length === 1 && container.children[0].textContent.includes('is not running')) {
          container.innerHTML = '';
        }

        logsRes.lines.forEach(item => {
          voiceLastLogId = Math.max(voiceLastLogId, item.id);
          const lineEl = colorizeVoiceLine(item.text);
          container.appendChild(lineEl);
        });

        // Auto scroll to bottom
        const parent = document.getElementById('voice-terminal');
        if (parent) {
          container.scrollTop = container.scrollHeight;
        }
      }
    }
  } catch (err) {
    // Ignore fetch errors during polling
  }
}

// Start Voice Assistant Polling loop (every 500ms for live responsive terminal)
setInterval(pollVoiceStatusAndLogs, 500);

// ─── Live Object Detection Toggle ─────────────────────────────────
let isDetectingObjects = false;

async function toggleObjectDetection() {
  const btn = document.getElementById('btn-toggle-detect');
  isDetectingObjects = !isDetectingObjects;
  
  try {
    const res = await api('toggle_detection', 'POST', { enable: isDetectingObjects });
    if (res && res.status === 'success') {
      isDetectingObjects = res.detect_objects;
      if (btn) {
        if (isDetectingObjects) {
          btn.textContent = '🎯 AI Vision: ON';
          btn.className = 'btn btn-primary';
          toast('Live YOLO object detection overlay enabled!', 'success');
        } else {
          btn.textContent = '🎯 AI Vision: OFF';
          btn.className = 'btn btn-outline';
          toast('Live object detection overlay disabled.', 'info');
        }
      }
    }
  } catch (err) {
    toast('Error toggling object detection', 'error');
  }
}

// ─── Logitech HD Camera Controls ──────────────────────────────────
let isLogiDetecting = false;
let logiStreamLive = false; // true once the first frame has loaded successfully

// Model is fixed to Grounding DINO — no switcher needed

async function updateWorldPrompts(classesStr) {
  try {
    const res = await api('set_detector_classes', 'POST', { classes: classesStr });
    if (res && res.status === 'success') {
      toast(`Updated detection targets: ${res.classes.join(', ')}`, 'success');
    }
  } catch (err) {
    toast('Error updating vision classes', 'error');
  }
}


async function toggleLogiDetection() {
  const btn = document.getElementById('btn-toggle-logi-detect');
  isLogiDetecting = !isLogiDetecting;
  try {
    const res = await api('toggle_logi_detection', 'POST', { enable: isLogiDetecting });
    if (res && res.status === 'success') {
      isLogiDetecting = res.detect_objects;
      if (btn) {
        btn.textContent = isLogiDetecting ? '🎯 AI Vision: ON' : '🎯 AI Vision: OFF';
        btn.className   = isLogiDetecting ? 'btn btn-primary' : 'btn btn-outline';
        btn.style.cssText = 'padding: 6px 12px; font-size: 0.85rem;';
        toast(isLogiDetecting ? 'Transformer AI Vision enabled!' : 'Transformer AI Vision disabled.', isLogiDetecting ? 'success' : 'info');
      }
    }
  } catch (err) {
    toast('Error toggling vision detection', 'error');
  }
}

// Called from img onload — marks stream as live and hides overlay permanently
function onLogiStreamLoaded() {
  logiStreamLive = true;
  const overlay = document.getElementById('logi-offline-overlay');
  if (overlay) overlay.style.display = 'none';
  const img = document.getElementById('logi-stream-img');
  if (img) img.style.opacity = '1';
  const dot = document.getElementById('logi-status-dot');
  const txt = document.getElementById('logi-status-text');
  if (dot) dot.style.background = '#10b981';
  if (txt) { txt.textContent = 'Live'; txt.style.color = '#10b981'; }
}

// Called from img onerror — resets flag and shows overlay
function onLogiStreamError(img) {
  logiStreamLive = false;
  const overlay = document.getElementById('logi-offline-overlay');
  if (overlay) overlay.style.display = 'flex';
  img.setAttribute('data-error', '1');
  const dot = document.getElementById('logi-status-dot');
  const txt = document.getElementById('logi-status-text');
  if (dot) dot.style.background = 'var(--text-muted)';
  if (txt) { txt.textContent = 'Reconnecting…'; txt.style.color = 'var(--text-muted)'; }
  setTimeout(() => {
    if (img.getAttribute('data-error') === '1') {
      img.src = '/api/logi_camera?t=' + Date.now();
      img.removeAttribute('data-error');
    }
  }, 2000);
}

function reconnectLogiStream() {
  logiStreamLive = false;
  const img = document.getElementById('logi-stream-img');
  if (img) {
    img.src = '/api/logi_camera?t=' + Date.now();
    toast('Reconnecting Logitech stream…', 'info');
  }
}

// Poll only FPS counter — never touch overlay visibility here
async function pollLogiStatus() {
  try {
    const s = await api('status');
    const fpsEl = document.getElementById('logi-fps-counter');
    if (s && s.logi_fps != null && s.logi_fps > 0) {
      if (fpsEl) fpsEl.textContent = s.logi_fps.toFixed(1) + ' FPS';
    }
  } catch (_) {}
}
setInterval(pollLogiStatus, 1000);

// ─── Sticker-Free Interactive XY Grid & Coordinate System ─────────
let showLogiGrid = true;
let logiOriginPx = { x: 960, y: 540 }; // Default origin (0,0) at image center (1920x1080)
let logiScaleMmPerPx = 0.35; // ~0.35mm per pixel at 55cm stand height

let lastClickedPx = null;
let lastClickedWorld = null;

function toggleLogiGrid() {
  showLogiGrid = !showLogiGrid;
  const btn = document.getElementById('btn-toggle-logi-grid');
  if (btn) {
    btn.textContent = showLogiGrid ? '🌐 Grid: ON' : '🌐 Grid: OFF';
    btn.className = showLogiGrid ? 'btn btn-outline' : 'btn btn-secondary';
  }
  drawLogiInteractiveGrid();
}

function pixelToWorldCoords(px, py) {
  // Origin (center of video stream): (960, 540)
  const raw_dx_mm = (px - 960) * logiScaleMmPerPx;
  const raw_dy_mm = (540 - py) * logiScaleMmPerPx;
  
  // Forward reach X: Moving UP on screen (py < 540, +raw_dy) reaches FORWARD (+X) into workspace
  const robot_x = 0.18 + (raw_dy_mm / 1000.0);
  
  // Left/Right Y: Moving RIGHT on screen (+raw_dx) moves to Robot Right (-Y)
  const robot_y = -(raw_dx_mm / 1000.0);
  
  return {
    raw_dx_mm,
    raw_dy_mm,
    robot_x,
    robot_y,
    x_m: robot_x,
    y_m: robot_y
  };
}

function updateLogiClickHUD() {
  const robotEl = document.getElementById('logi-click-coords-robot');
  if (lastClickedWorld && robotEl) {
    robotEl.textContent = `X: ${lastClickedWorld.robot_x.toFixed(3)}m,  Y: ${lastClickedWorld.robot_y.toFixed(3)}m`;
  }
}

function setOriginToCurrentClick() {
  if (!lastClickedPx) {
    toast('Click somewhere on the video stream first!', 'info');
    return;
  }
  logiOriginPx = { x: lastClickedPx.x, y: lastClickedPx.y };
  lastClickedWorld = pixelToWorldCoords(lastClickedPx.x, lastClickedPx.y);
  updateLogiClickHUD();
  drawLogiInteractiveGrid();
  toast('Robot Base Origin (0,0) calibrated to clicked spot!', 'success');
}

function resetOriginToCenter() {
  logiOriginPx = { x: 960, y: 540 };
  if (lastClickedPx) {
    lastClickedWorld = pixelToWorldCoords(lastClickedPx.x, lastClickedPx.y);
    updateLogiClickHUD();
  }
  drawLogiInteractiveGrid();
  toast('Origin reset back to image center (960, 540).', 'info');
}

function sendClickedPointToMoveTab() {
  if (!lastClickedWorld) return;

  const inputX = document.getElementById('input-x');
  const inputY = document.getElementById('input-y');

  if (inputX && inputY) {
    inputX.value = lastClickedWorld.robot_x.toFixed(3);
    inputY.value = lastClickedWorld.robot_y.toFixed(3);
    toast(`Inverted & sent coordinates to Move tab: X=${lastClickedWorld.robot_x.toFixed(3)}m, Y=${lastClickedWorld.robot_y.toFixed(3)}m`, 'success');
    switchTab('move');
  }
}

function getCanvasPointerPixel(e, canvas) {
  const rect = canvas.getBoundingClientRect();
  const containerW = rect.width;
  const containerH = rect.height;
  const canvasW = canvas.width || 1920;
  const canvasH = canvas.height || 1080;

  const containerRatio = containerW / containerH;
  const nativeRatio = canvasW / canvasH;

  let renderW, renderH, offsetX, offsetY;
  if (containerRatio > nativeRatio) {
    renderH = containerH;
    renderW = containerH * nativeRatio;
    offsetX = (containerW - renderW) / 2;
    offsetY = 0;
  } else {
    renderW = containerW;
    renderH = containerW / nativeRatio;
    offsetX = 0;
    offsetY = (containerH - renderH) / 2;
  }

  const mouseX = e.clientX - rect.left - offsetX;
  const mouseY = e.clientY - rect.top - offsetY;

  const clampedX = Math.max(0, Math.min(renderW, mouseX));
  const clampedY = Math.max(0, Math.min(renderH, mouseY));

  const px = Math.round((clampedX / renderW) * canvasW);
  const py = Math.round((clampedY / renderH) * canvasH);
  return { px, py };
}

function initLogiGridCanvas() {
  const canvas = document.getElementById('logi-grid-canvas');
  if (!canvas) return;

  canvas.addEventListener('pointerdown', (e) => {
    const { px, py } = getCanvasPointerPixel(e, canvas);

    lastClickedPx = { x: px, y: py };
    lastClickedWorld = pixelToWorldCoords(px, py);

    // If in calibration picking mode, capture pixel and stop
    if (typeof calibCaptureClick === 'function' && calibCaptureClick(px, py)) {
      drawLogiInteractiveGrid();
      return;
    }

    updateLogiClickHUD();
    drawLogiInteractiveGrid();

    // Enable Send to Move Tab button
    const sendBtn = document.getElementById('btn-send-click-move');
    if (sendBtn) sendBtn.disabled = false;

    // Direct Instant Move using calibration-aware coordinates
    if (isConnected) {
      getTargetCoords(px, py).then(coords => {
        const label = coords.calibrated ? '🎯 [calibrated]' : '⚠️ [approx]';
        toast(`${label} Moving to X: ${coords.robot_x.toFixed(3)}m, Y: ${coords.robot_y.toFixed(3)}m...`, 'info');
        api('move_xyz', 'POST', {
          x: coords.robot_x,
          y: coords.robot_y,
          z: tableZ,
          duration: 2.5
        }).then(res => {
          if (res.status === 'moving') {
            toast(`✓ Moving to (${coords.robot_x.toFixed(2)}m, ${coords.robot_y.toFixed(2)}m)`, 'success');
          } else if (res.message) {
            toast(res.message, 'error');
          }
        }).catch(console.error);
      });
    }
  });

  drawLogiInteractiveGrid();
}

function drawLogiInteractiveGrid() {
  const canvas = document.getElementById('logi-grid-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  if (showLogiGrid) {
    // 1. Draw Grid Lines (Every 50mm / 5cm)
    const stepMm = 50;
    const stepPx = stepMm / logiScaleMmPerPx;

    ctx.strokeStyle = 'rgba(0, 229, 255, 0.12)';
    ctx.lineWidth = 1;

    for (let x = logiOriginPx.x % stepPx; x < w; x += stepPx) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
    for (let y = logiOriginPx.y % stepPx; y < h; y += stepPx) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    // 2. Draw Main Axes (Origin)
    ctx.beginPath(); ctx.moveTo(0, logiOriginPx.y); ctx.lineTo(w, logiOriginPx.y);
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.7)'; ctx.lineWidth = 2; ctx.stroke();

    ctx.beginPath(); ctx.moveTo(logiOriginPx.x, 0); ctx.lineTo(logiOriginPx.x, h);
    ctx.strokeStyle = 'rgba(16, 185, 129, 0.7)'; ctx.lineWidth = 2; ctx.stroke();

    ctx.beginPath();
    ctx.arc(logiOriginPx.x, logiOriginPx.y, 6, 0, Math.PI * 2);
    ctx.fillStyle = '#00e5ff'; ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();

    ctx.font = '600 13px Inter, sans-serif';
    ctx.fillStyle = '#00e5ff';
    ctx.fillText('Origin (0,0)', logiOriginPx.x + 10, logiOriginPx.y - 10);
  }

  // Draw Saved Calibration Points (P1, P2, P3...)
  if (calibPoints && calibPoints.length > 0) {
    calibPoints.forEach((p, i) => {
      ctx.save();
      ctx.beginPath();
      ctx.arc(p.px, p.py, 12, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255, 153, 0, 0.2)';
      ctx.fill();
      ctx.strokeStyle = '#ff9900';
      ctx.lineWidth = 2;
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(p.px, p.py, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#ff9900';
      ctx.fill();

      ctx.font = 'bold 12px monospace';
      ctx.fillStyle = '#ff9900';
      ctx.fillText(`P${i+1}`, p.px + 14, p.py - 4);
      ctx.restore();
    });
  }

  // Draw Target Reticle at Last Clicked Location
  if (lastClickedPx) {
    const cx = lastClickedPx.x;
    const cy = lastClickedPx.y;

    // Glowing target ring
    ctx.beginPath();
    ctx.arc(cx, cy, 18, 0, Math.PI * 2);
    ctx.strokeStyle = '#ff9900';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(cx, cy, 5, 0, Math.PI * 2);
    ctx.fillStyle = '#ff9900';
    ctx.fill();

    // Crosshair ticks
    ctx.strokeStyle = '#ff9900';
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(cx - 28, cy); ctx.lineTo(cx - 10, cy); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx + 10, cy); ctx.lineTo(cx + 28, cy); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx, cy - 28); ctx.lineTo(cx, cy - 10); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx, cy + 10); ctx.lineTo(cx, cy + 28); ctx.stroke();

    // HUD Label Box
    if (lastClickedWorld) {
      const text = `Robot: X=${lastClickedWorld.robot_x.toFixed(3)}m  Y=${lastClickedWorld.robot_y.toFixed(3)}m`;
      ctx.font = 'bold 14px monospace';
      const textWidth = ctx.measureText(text).width;

      const badgeX = cx + 22;
      const badgeY = cy - 22;

      ctx.fillStyle = 'rgba(0, 0, 0, 0.75)';
      ctx.strokeStyle = 'rgba(255, 153, 0, 0.8)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect(badgeX, badgeY - 18, textWidth + 16, 26, 6);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = '#ff9900';
      ctx.fillText(text, badgeX + 8, badgeY);
    }
  }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
  initLogiGridCanvas();
  pollCalibrationStatus();
  updateTableZDisplay();
  pollDetectedObjects();
});
setTimeout(initLogiGridCanvas, 1000);


// ─── Camera Calibration System ────────────────────────────────────

let calibIsCalibrated = false;
let calibPendingPx = null;   // pixel coords of last click while adding point
let calibPoints = [];        // local mirror of server calibration points
let calibPickingMode = false; // true when waiting for user to click video

async function pollCalibrationStatus() {
  try {
    const res = await api('calibration/status');
    calibIsCalibrated = res.calibrated;
    calibPoints = res.points || [];
    updateCalibStatusBadge();
  } catch (_) {}
  setTimeout(pollCalibrationStatus, 5000);
}

function updateCalibStatusBadge() {
  const dot = document.getElementById('calib-status-dot');
  const txt = document.getElementById('calib-status-text');
  if (!dot || !txt) return;
  if (calibIsCalibrated) {
    dot.style.background = '#10b981';
    txt.textContent = `Calibrated (${calibPoints.length} pts)`;
    txt.style.color = '#10b981';
  } else if (calibPoints.length > 0) {
    dot.style.background = '#f59e0b';
    txt.textContent = `${calibPoints.length} pts — not yet computed`;
    txt.style.color = '#f59e0b';
  } else {
    dot.style.background = '#6b7280';
    txt.textContent = 'No Calibration';
    txt.style.color = 'var(--text-muted)';
  }
}

function openCalibrationPanel() {
  const panel = document.getElementById('calib-panel');
  if (panel) {
    panel.style.display = 'flex';
    switchCalibTab('twoobjects');
  }
}

function closeCalibrationPanel() {
  const panel = document.getElementById('calib-panel');
  if (panel) panel.style.display = 'none';
  isCropBoxMode = false;
  calibPickingMode = false;
  calibPendingPx = null;
  const pend = document.getElementById('calib-pending');
  if (pend) pend.style.display = 'none';
  const newForm = document.getElementById('calib-new-point-form');
  if (newForm) newForm.style.display = 'none';
  drawLogiInteractiveGrid();
}

function renderCalibPointsList() {
  const list = document.getElementById('calib-points-list');
  if (!list) return;
  if (calibPoints.length === 0) {
    list.innerHTML = '<div style="color: var(--text-muted); font-size: 0.8rem; padding: 4px 0;">No calibration points yet. Click a marker on the video, then click ➕ Add Point.</div>';
    document.getElementById('btn-compute-calib').disabled = true;
    return;
  }
  list.innerHTML = calibPoints.map((p, i) => `
    <div style="display:flex; align-items:center; gap:10px; padding:5px 10px; background:rgba(255,255,255,0.04); border-radius:6px; font-size:0.8rem; font-family:monospace;">
      <span style="color:#ff9900; font-weight:700; min-width:22px;">P${i+1}</span>
      <span style="color:var(--text-muted);">pixel(${Math.round(p.px)}, ${Math.round(p.py)})</span>
      <span style="color:#10b981;">→ robot(${p.rx.toFixed(3)}m, ${p.ry.toFixed(3)}m)</span>
      <button onclick="calibRemovePoint(${i})" style="margin-left:auto; background:none; border:none; color:#ef4444; cursor:pointer; font-size:0.85rem;">✕</button>
    </div>
  `).join('');
  document.getElementById('btn-compute-calib').disabled = calibPoints.length < 4;
}

function calibAddPoint() {
  // Enter picking mode: next click on canvas will be captured as calib pixel
  calibPickingMode = true;
  calibPendingPx = null;
  document.getElementById('calib-pending').style.display = 'block';
  document.getElementById('calib-new-point-form').style.display = 'none';
  toast('Click on a marker in the video stream to set its pixel position', 'info');
}

// Called by the canvas click handler when in picking mode
function calibCaptureClick(px, py) {
  if (!calibPickingMode) return false;
  calibPendingPx = { x: px, y: py };
  calibPickingMode = false;
  document.getElementById('calib-pending').style.display = 'none';
  document.getElementById('calib-new-point-form').style.display = 'flex';
  toast(`Pixel (${px}, ${py}) captured — enter robot coordinates below`, 'success');
  return true; // consumed
}

async function calibConfirmPoint() {
  if (!calibPendingPx) { toast('Click a point on the video first!', 'error'); return; }
  const rx = parseFloat(document.getElementById('calib-input-rx').value);
  const ry = parseFloat(document.getElementById('calib-input-ry').value);
  if (isNaN(rx) || isNaN(ry)) { toast('Enter valid robot coordinates', 'error'); return; }
  const res = await api('calibration/add_point', 'POST', {
    px: calibPendingPx.x, py: calibPendingPx.y,
    rx, ry, index: calibPoints.length
  });
  calibPoints = res.points || [];
  calibPendingPx = null;
  document.getElementById('calib-new-point-form').style.display = 'none';
  renderCalibPointsList();
  updateCalibStatusBadge();
  drawLogiInteractiveGrid();
  toast(`Point P${calibPoints.length} added!`, 'success');
}

function calibCancelPoint() {
  calibPendingPx = null;
  calibPickingMode = false;
  document.getElementById('calib-pending').style.display = 'none';
  document.getElementById('calib-new-point-form').style.display = 'none';
}

async function calibReadArmPosition() {
  const btn = event.target;
  const origText = btn.textContent;
  btn.textContent = '⏳ Reading...';
  btn.disabled = true;

  try {
    const res = await api('arm_position');
    if (res.status === 'ok') {
      document.getElementById('calib-input-rx').value = res.x.toFixed(4);
      document.getElementById('calib-input-ry').value = res.y.toFixed(4);
      toast(`✅ Arm position read: X=${res.x.toFixed(3)}m, Y=${res.y.toFixed(3)}m, Z=${res.z.toFixed(3)}m`, 'success');
    } else {
      toast(`Could not read arm: ${res.message}`, 'error');
    }
  } catch (e) {
    toast('Failed to read arm position — is the arm connected?', 'error');
  }

  btn.textContent = origText;
  btn.disabled = false;
}

async function setTableHeight() {
  const btn = event.target;
  const origText = btn.textContent;
  btn.textContent = '⏳ Reading...';
  btn.disabled = true;

  try {
    const res = await api('arm_position');
    if (res.status === 'ok') {
      tableZ = res.z;
      localStorage.setItem('tableZ', tableZ.toFixed(5));
      updateTableZDisplay();
      toast(`✅ Table Z set to ${tableZ.toFixed(4)}m — robot will now touch the table on every click!`, 'success');
    } else {
      toast(`Could not read arm: ${res.message}`, 'error');
    }
  } catch (e) {
    toast('Failed to read arm position — is the arm connected?', 'error');
  }

  btn.textContent = origText;
  btn.disabled = false;
}

function adjustTableZ(deltaM) {
  tableZ = parseFloat((tableZ + deltaM).toFixed(4));
  localStorage.setItem('tableZ', tableZ.toFixed(5));
  updateTableZDisplay();
  toast(`Table Z adjusted to ${tableZ.toFixed(4)}m`, 'info');
}

function updateTableZDisplay() {
  const el = document.getElementById('table-z-display');
  if (el) el.textContent = `Z: ${tableZ.toFixed(4)}m`;
}

async function calibRemovePoint(idx) {
  const res = await api('calibration/remove_point', 'POST', { index: idx });
  calibPoints = res.points || [];
  calibIsCalibrated = false;
  renderCalibPointsList();
  updateCalibStatusBadge();
  drawLogiInteractiveGrid();
}

async function calibCompute() {
  if (calibPoints.length < 4) { toast('Need at least 4 calibration points!', 'error'); return; }
  const btn = document.getElementById('btn-compute-calib');
  btn.disabled = true;
  btn.textContent = '⏳ Computing...';
  try {
    const res = await api('calibration/compute', 'POST');
    if (res.status === 'ok') {
      calibIsCalibrated = true;
      updateCalibStatusBadge();
      toast('✅ Calibration computed! Click-to-move is now accurate.', 'success');
      closeCalibrationPanel();
    } else {
      toast(`Calibration failed: ${res.message}`, 'error');
    }
  } catch (e) {
    toast('Calibration computation error', 'error');
  }
  btn.disabled = false;
  btn.textContent = '✓ Compute Calibration';
}

async function calibReset() {
  await api('calibration/reset', 'POST');
  calibPoints = [];
  calibIsCalibrated = false;
  renderCalibPointsList();
  updateCalibStatusBadge();
  drawLogiInteractiveGrid();
  toast('Calibration reset.', 'info');
}

// ─── Manual Calibration Complete ───────────────────────────────────

// Calibration-aware coordinate transform: uses homography if calibrated, else fallback
async function getTargetCoords(px, py) {
  if (calibIsCalibrated) {
    try {
      const res = await api('calibration/transform', 'POST', { px, py });
      if (res.status === 'ok') return { robot_x: res.rx, robot_y: res.ry, calibrated: true };
    } catch (_) {}
  }
  // Fallback: simple Y-inversion approximation
  const coords = pixelToWorldCoords(px, py);
  return { robot_x: coords.robot_x, robot_y: coords.robot_y, calibrated: false };
}

// ─── Live Detected Objects on Table Grid ───────────────────────────

let detectedObjectsList = [];

async function pollDetectedObjects() {
  if (isLogiDetecting) {
    try {
      const res = await api('detected_objects');
      if (res && res.status === 'ok') {
        detectedObjectsList = res.objects || [];
        renderDetectedObjectsList();
      }
    } catch (_) {}
  } else {
    const container = document.getElementById('logi-detected-objects-container');
    if (container && container.style.display !== 'none') {
      container.style.display = 'none';
    }
  }
  setTimeout(pollDetectedObjects, 600);
}

function renderDetectedObjectsList() {
  const container = document.getElementById('logi-detected-objects-container');
  const countEl = document.getElementById('detected-objects-count');
  const listEl = document.getElementById('detected-objects-list');
  if (!container || !countEl || !listEl) return;

  if (detectedObjectsList.length === 0) {
    container.style.display = isLogiDetecting ? 'flex' : 'none';
    countEl.textContent = '0 found';
    listEl.innerHTML = '<span style="font-size:0.78rem; color:var(--text-muted); padding:4px 0;">Searching for objects in camera view...</span>';
    return;
  }

  container.style.display = 'flex';
  countEl.textContent = `${detectedObjectsList.length} found`;

  listEl.innerHTML = detectedObjectsList.map((obj, i) => {
    const hasCoords = obj.robot_x !== null && obj.robot_y !== null;
    const coordsBadge = hasCoords
      ? `<span style="color:#10b981; font-weight:700; font-family:monospace; font-size:0.8rem;">X: ${obj.robot_x.toFixed(3)}m, Y: ${obj.robot_y.toFixed(3)}m</span>`
      : `<span style="color:#f59e0b; font-size:0.75rem;">(Calibrate for Grid (X,Y))</span>`;

    return `
      <div style="display:flex; align-items:center; gap:8px; padding:6px 12px; background:rgba(255,255,255,0.04); border:1px solid rgba(0,229,255,0.25); border-radius:6px; font-size:0.82rem;">
        <span style="color:#00e5ff; font-weight:700;">📦 ${obj.name}</span>
        <span style="color:var(--text-muted); font-size:0.75rem;">(${Math.round(obj.confidence * 100)}%)</span>
        <span style="border-left:1px solid rgba(255,255,255,0.1); padding-left:8px;">${coordsBadge}</span>
        ${hasCoords ? `
          <button onclick="moveArmToObject(${obj.robot_x}, ${obj.robot_y}, '${obj.name}')" class="btn btn-accent" style="padding:3px 10px; font-size:0.75rem; margin-left:4px;" title="Send robot arm to this exact object">🚀 Move Here</button>
        ` : ''}
      </div>
    `;
  }).join('');
}

async function moveArmToObject(rx, ry, name) {
  if (!isConnected) {
    toast('Click [🦾 Connect Follower] above to enable arm movement', 'info');
    return;
  }
  toast(`🎯 Moving arm to ${name || 'object'} at X:${rx.toFixed(3)}m, Y:${ry.toFixed(3)}m...`, 'info');
  try {
    const res = await api('move_xyz', 'POST', {
      x: rx,
      y: ry,
      z: tableZ,
      duration: 2.5
    });
    if (res && res.status === 'moving') {
      toast(`✓ Arm moving to ${name || 'object'} (${rx.toFixed(2)}m, ${ry.toFixed(2)}m)`, 'success');
    } else if (res && res.message) {
      toast(res.message, 'error');
    }
  } catch (err) {
    console.error(err);
    toast('Error sending move command', 'error');
  }
}


// ═══════════════════════════════════════════════════════════════════
//  AI TRAINING & AUTONOMOUS POLICY CONTROLLER
// ═══════════════════════════════════════════════════════════════════

let aiCurrentSubTab = 'record';
let isAIRecording = false;
let aiRecordingTimer = null;

function switchAISubTab(tab) {
  aiCurrentSubTab = tab;
  const tabs = ['record', 'train', 'eval'];
  tabs.forEach(t => {
    const btn = document.getElementById(`ai-subtab-${t}`);
    const panel = document.getElementById(`ai-panel-${t}`);
    if (btn) btn.className = (t === tab) ? 'btn btn-accent' : 'btn btn-outline';
    if (panel) panel.style.display = (t === tab) ? 'flex' : 'none';
  });

  if (tab === 'record') {
    loadAIDatasetInfo();
  } else if (tab === 'train') {
    loadAIDatasetSelect();
    pollAITrainingStatus();
  } else if (tab === 'eval') {
    loadAIModelsList();
    pollAIEvalStatus();
  }
}

// ─── 1. Episode Recording Controller ───────────────────────────────

function onAIDatasetChanged() {
  loadAIDatasetInfo();
}

async function loadAIDatasetInfo() {
  const dsInput = document.getElementById('ai-input-dataset-name');
  const dsName = dsInput ? dsInput.value.trim() : 'pick_ball_so101';
  if (!dsName) return;

  try {
    const res = await api(`ai/dataset/info?dataset=${encodeURIComponent(dsName)}`);
    if (res && res.status === 'ok' && res.info) {
      const count = res.info.count || 0;
      const countBadge = document.getElementById('ai-rec-count-badge');
      if (countBadge) {
        countBadge.textContent = `Episodes: ${count} / 30`;
        countBadge.style.color = count >= 30 ? '#10b981' : '#ff9900';
      }

      const tbody = document.getElementById('ai-episodes-list-body');
      if (tbody) {
        if (!res.info.episodes || res.info.episodes.length === 0) {
          tbody.innerHTML = '<tr><td colspan="5" style="padding: 24px; text-align: center; color: var(--text-muted);">No episodes recorded yet in this dataset. Click Start Recording above!</td></tr>';
        } else {
          tbody.innerHTML = res.info.episodes.map(ep => `
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.04);">
              <td style="padding: 8px 12px; font-weight: 700; color: #00e5ff;">#${ep.index + 1}</td>
              <td style="padding: 8px 12px; cursor: pointer;" onclick="showEpisodeVideo(${ep.index})">
                ${ep.thumb ? `
                  <div style="position: relative; width: 68px; height: 38px; border-radius: 4px; overflow: hidden; border: 1px solid var(--border); display: inline-block;">
                    <img src="${ep.thumb}" style="width: 100%; height: 100%; object-fit: cover;">
                    <div style="position: absolute; inset: 0; background: rgba(0,0,0,0.4); display: flex; align-items: center; justify-content: center; font-size: 0.75rem; color: #fff;">▶</div>
                  </div>
                ` : '<span style="color:var(--text-muted);">—</span>'}
              </td>
              <td style="padding: 8px 12px; font-family: monospace;">${ep.duration_s}s</td>
              <td style="padding: 8px 12px; font-family: monospace;">${ep.steps} frames</td>
              <td style="padding: 8px 12px; white-space: nowrap;">
                <button class="btn btn-outline" onclick="showEpisodeVideo(${ep.index})" style="padding: 3px 8px; font-size: 0.75rem; margin-right: 4px; border-color: #00e5ff; color: #00e5ff;">🎬 Watch Video</button>
                <button class="btn btn-outline" onclick="replayAIEpisode(${ep.index})" style="padding: 3px 8px; font-size: 0.75rem; margin-right: 4px;">🦾 Replay Arm</button>
                <button class="btn btn-secondary" onclick="deleteAIEpisode(${ep.index})" style="padding: 3px 8px; font-size: 0.75rem;">🗑</button>
              </td>
            </tr>
          `).join('');
        }
      }
    }
  } catch (_) {}
}

async function startAIEpisode() {
  if (!isConnected) {
    toast('Please click [🦾 Connect Follower] or [Connect Both] first', 'info');
  }
  const dsInput = document.getElementById('ai-input-dataset-name');
  const taskInput = document.getElementById('ai-input-task-desc');
  const dataset = dsInput ? dsInput.value.trim() : 'pick_ball_so101';
  const task = taskInput ? taskInput.value.trim() : 'Pick up the ball from the desk';

  try {
    const res = await api('ai/dataset/start_episode', 'POST', { dataset, task });
    if (res && res.status === 'recording') {
      isAIRecording = true;
      updateAIRecordingUI(true);
      toast('🔴 Recording & Teleop Active! Move leader arm to pick up the ball.', 'info');
      pollAIRecordingTimer();
    }
  } catch (e) {
    toast('Failed to start episode recording', 'error');
  }
}

async function stopAIEpisode(save) {
  try {
    const res = await api('ai/dataset/stop_episode', 'POST', { save });
    isAIRecording = false;
    updateAIRecordingUI(false);

    if (save && res && res.status === 'saved') {
      toast(`💾 Saved Episode #${res.episode_index + 1} (${res.steps} frames)!`, 'success');
      loadAIDatasetInfo();
    } else if (!save) {
      toast('🗑 Episode discarded.', 'info');
    } else if (res && res.message) {
      toast(res.message, 'error');
    }
  } catch (e) {
    toast('Error stopping episode recording', 'error');
    isAIRecording = false;
    updateAIRecordingUI(false);
  }
}

function updateAIRecordingUI(recording) {
  const startBtn = document.getElementById('btn-ai-start-rec');
  const saveBtn = document.getElementById('btn-ai-save-rec');
  const discardBtn = document.getElementById('btn-ai-discard-rec');
  const timerBadge = document.getElementById('ai-rec-timer-badge');

  if (startBtn) startBtn.style.display = recording ? 'none' : 'inline-block';
  if (saveBtn) saveBtn.style.display = recording ? 'inline-block' : 'none';
  if (discardBtn) discardBtn.style.display = recording ? 'inline-block' : 'none';
  if (timerBadge) {
    timerBadge.style.color = recording ? '#ef4444' : 'var(--text-muted)';
    timerBadge.style.borderColor = recording ? 'rgba(239,68,68,0.5)' : 'var(--border)';
    if (!recording) timerBadge.textContent = '0.0s (0 frames)';
  }
}

async function pollAIRecordingTimer() {
  if (!isAIRecording) return;
  try {
    const res = await api('ai/dataset/status');
    if (res && res.status === 'ok' && res.is_recording) {
      const timerBadge = document.getElementById('ai-rec-timer-badge');
      if (timerBadge) {
        timerBadge.textContent = `⏱ ${res.duration_s}s (${res.current_steps} frames)`;
      }
      setTimeout(pollAIRecordingTimer, 200);
    } else {
      isAIRecording = false;
      updateAIRecordingUI(false);
    }
  } catch (_) {
    setTimeout(pollAIRecordingTimer, 500);
  }
}

let currentVideoModalIndex = 0;
let currentVideoSpeed = 1.0;

function showEpisodeVideo(index) {
  currentVideoModalIndex = index;
  const dsInput = document.getElementById('ai-input-dataset-name');
  const dataset = dsInput ? dsInput.value.trim() : 'pick_ball_so101';

  const modal = document.getElementById('ai-video-modal');
  const title = document.getElementById('ai-video-modal-title');
  const streamImg = document.getElementById('ai-video-modal-stream');
  const armReplayBtn = document.getElementById('btn-modal-arm-replay');

  if (title) title.textContent = `🎬 Episode #${index + 1} Video Playback (${dataset})`;
  if (streamImg) {
    streamImg.src = `/api/ai/dataset/video?dataset=${dataset}&index=${index}&speed=${currentVideoSpeed}&loop=true&t=${Date.now()}`;
  }
  if (armReplayBtn) {
    armReplayBtn.onclick = () => replayAIEpisode(index);
  }
  if (modal) modal.style.display = 'flex';
}

function setEpisodeVideoSpeed(speed) {
  currentVideoSpeed = speed;
  const dsInput = document.getElementById('ai-input-dataset-name');
  const dataset = dsInput ? dsInput.value.trim() : 'pick_ball_so101';
  const streamImg = document.getElementById('ai-video-modal-stream');
  if (streamImg) {
    streamImg.src = `/api/ai/dataset/video?dataset=${dataset}&index=${currentVideoModalIndex}&speed=${speed}&loop=true&t=${Date.now()}`;
  }
  toast(`Speed set to ${speed}x`, 'info');
}

function closeEpisodeVideoModal() {
  const modal = document.getElementById('ai-video-modal');
  const streamImg = document.getElementById('ai-video-modal-stream');
  if (streamImg) streamImg.src = '';
  if (modal) modal.style.display = 'none';
}

async function replayAIEpisode(index) {
  const dsInput = document.getElementById('ai-input-dataset-name');
  const dataset = dsInput ? dsInput.value.trim() : 'pick_ball_so101';
  toast(`▶ Replaying Episode #${index + 1} on follower arm...`, 'info');
  try {
    const res = await api('ai/dataset/replay_episode', 'POST', { dataset, index });
    if (res && res.status === 'replaying') {
      toast(`✓ Replaying Episode #${index + 1}`, 'success');
    }
  } catch (e) {
    toast('Error triggering episode replay', 'error');
  }
}

async function deleteAIEpisode(index) {
  const dsInput = document.getElementById('ai-input-dataset-name');
  const dataset = dsInput ? dsInput.value.trim() : 'pick_ball_so101';
  if (!confirm(`Delete Episode #${index + 1}?`)) return;

  try {
    const res = await api('ai/dataset/delete_episode', 'POST', { dataset, index });
    if (res && res.status === 'ok') {
      toast(`✓ Deleted Episode #${index + 1}. Library updated!`, 'info');
      await loadAIDatasetInfo();
      await loadAIDatasetSelect();
    } else {
      toast('Failed to delete episode', 'error');
    }
  } catch (e) {
    toast('Error deleting episode', 'error');
  }
}

// Global Spacebar Shortcut for fast continuous episode collection
document.addEventListener('keydown', (e) => {
  if (e.code === 'Space' && activeTab === 'ai' && aiCurrentSubTab === 'record') {
    if (document.activeElement && (document.activeElement.tagName === 'INPUT' || document.activeElement.tagName === 'TEXTAREA')) {
      return;
    }
    e.preventDefault();
    if (!isAIRecording) {
      startAIEpisode();
    } else {
      stopAIEpisode(true);
    }
  }
});


// ─── 2. Model Training Controller ──────────────────────────────────

async function loadAIDatasetSelect() {
  try {
    const res = await api('ai/dataset/list');
    const select = document.getElementById('ai-train-select-dataset');
    if (select && res && res.datasets) {
      if (res.datasets.length === 0) {
        select.innerHTML = '<option value="pick_ball_so101">pick_ball_so101 (0 episodes)</option>';
      } else {
        select.innerHTML = res.datasets.map(d => `
          <option value="${d.name}">${d.name} (${d.episodes} episodes)</option>
        `).join('');
      }
    }
  } catch (_) {}
}

async function startAITraining() {
  const dsSelect = document.getElementById('ai-train-select-dataset');
  const dataset = dsSelect ? dsSelect.value : 'pick_ball_so101';
  const steps = parseInt(document.getElementById('ai-train-steps')?.value || '25000', 10);
  const device = document.getElementById('ai-train-device')?.value || 'mps';
  const batch_size = parseInt(document.getElementById('ai-train-batch')?.value || '8', 10);

  const startBtn = document.getElementById('btn-start-train');
  const stopBtn = document.getElementById('btn-stop-train');
  if (startBtn) startBtn.disabled = true;
  if (stopBtn) stopBtn.disabled = false;

  toast(`🚀 Starting ACT training (${steps.toLocaleString()} steps) on ${device.toUpperCase()}...`, 'info');

  try {
    const res = await api('ai/train/start', 'POST', { dataset, steps, device, batch_size });
    if (res && res.status === 'started') {
      toast('⚡ Training launched! Monitoring live loss & ETA...', 'success');
      pollAITrainingStatus();
    } else if (res && res.message) {
      toast(res.message, 'error');
      if (startBtn) startBtn.disabled = false;
      if (stopBtn) stopBtn.disabled = true;
    }
  } catch (e) {
    toast('Error starting training', 'error');
    if (startBtn) startBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = true;
  }
}

async function stopAITraining() {
  await api('ai/train/stop', 'POST');
  toast('⏹ Training stopped.', 'info');
  const startBtn = document.getElementById('btn-start-train');
  const stopBtn = document.getElementById('btn-stop-train');
  if (startBtn) startBtn.disabled = false;
  if (stopBtn) stopBtn.disabled = true;
}

async function pollAITrainingStatus() {
  try {
    const res = await api('ai/train/status');
    if (!res) return;

    const startBtn = document.getElementById('btn-start-train');
    const stopBtn = document.getElementById('btn-stop-train');
    const stepLabel = document.getElementById('ai-train-step-label');
    const lossBadge = document.getElementById('ai-train-loss-badge');
    const speedBadge = document.getElementById('ai-train-speed-badge');
    const etaBadge = document.getElementById('ai-train-eta-badge');
    const progressBar = document.getElementById('ai-train-progress-bar');
    const term = document.getElementById('ai-train-terminal');

    if (startBtn) startBtn.disabled = res.is_training;
    if (stopBtn) stopBtn.disabled = !res.is_training;

    if (res.total_steps > 0) {
      const pct = Math.min(100, Math.round((res.step / res.total_steps) * 100));
      if (progressBar) progressBar.style.width = `${pct}%`;
      if (stepLabel) stepLabel.textContent = `Step: ${res.step.toLocaleString()} / ${res.total_steps.toLocaleString()} (${pct}%)`;
    }

    if (lossBadge) lossBadge.textContent = `Loss: ${res.loss ? res.loss.toFixed(4) : '--'}`;
    if (speedBadge) speedBadge.textContent = `Speed: ${res.fps ? res.fps.toFixed(1) : '--'} FPS`;
    if (etaBadge) etaBadge.textContent = `ETA: ${res.eta || '--:--:--'}`;

    if (term && res.logs && res.logs.length > 0) {
      term.textContent = res.logs.join('\n');
      term.scrollTop = term.scrollHeight;
    }

    if (res.is_training) {
      setTimeout(pollAITrainingStatus, 1000);
    }
  } catch (_) {
    setTimeout(pollAITrainingStatus, 2000);
  }
}


// ─── 3. Autonomous Ball Picker Inference Controller ────────────────

async function loadAIModelsList() {
  try {
    const res = await api('ai/models/list');
    const select = document.getElementById('ai-eval-select-model');
    if (select && res && res.models) {
      if (res.models.length === 0) {
        select.innerHTML = '<option value="">No trained models found yet</option>';
      } else {
        select.innerHTML = res.models.map(m => `
          <option value="${m.path}">${m.name}</option>
        `).join('');
      }
    }
  } catch (_) {}
}

async function startAIEval() {
  const select = document.getElementById('ai-eval-select-model');
  const model_path = select ? select.value : '';
  if (!model_path) {
    toast('Select a trained model checkpoint first!', 'error');
    return;
  }

  const startBtn = document.getElementById('btn-start-eval');
  const stopBtn = document.getElementById('btn-stop-eval');
  if (startBtn) startBtn.disabled = true;
  if (stopBtn) stopBtn.disabled = false;

  toast('🚀 Running autonomous ball picker policy...', 'info');

  try {
    const res = await api('ai/eval/start', 'POST', { model_path });
    if (res && res.status === 'started') {
      toast('🤖 Autonomous policy active at 30 FPS!', 'success');
      pollAIEvalStatus();
    } else if (res && res.message) {
      toast(res.message, 'error');
      if (startBtn) startBtn.disabled = false;
      if (stopBtn) stopBtn.disabled = true;
    }
  } catch (e) {
    toast('Error starting autonomous evaluation', 'error');
    if (startBtn) startBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = true;
  }
}

async function stopAIEval() {
  await api('ai/eval/stop', 'POST');
  toast('⏹ Autonomous policy stopped.', 'info');
  const startBtn = document.getElementById('btn-start-eval');
  const stopBtn = document.getElementById('btn-stop-eval');
  if (startBtn) startBtn.disabled = false;
  if (stopBtn) stopBtn.disabled = true;
}

async function returnToStartingPose() {
  const btn = document.getElementById('btn-eval-home');
  if (btn) btn.disabled = true;
  toast('🏠 Moving arm back to initial starting / ready pose...', 'info');

  try {
    const res = await api('home', 'POST');
    if (res && res.status === 'moving_home') {
      toast('✅ Arm returned to starting pose!', 'success');
    } else if (res && res.message) {
      toast(res.message, 'warning');
    }
  } catch (e) {
    toast('Error returning to starting pose', 'error');
  } finally {
    if (btn) btn.disabled = false;
  }
}

function renderTrajectoryWaveform(predChunk) {
  const canvas = document.getElementById('ai-eval-trajectory-canvas');
  if (!canvas || !predChunk || predChunk.length === 0) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  // Background Grid
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
  ctx.lineWidth = 1;
  for (let y = 0; y < h; y += 30) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  // Colors for key joints (Pan, Lift, Elbow, Gripper)
  const colors = ['#38bdf8', '#f43f5e', '#a855f7', '#fbbf24', '#e2e8f0', '#4ade80'];
  const labels = ['Pan', 'Lift', 'Elbow', 'WFlex', 'WRoll', 'Gripper'];

  const numSteps = predChunk.length;
  const stepWidth = w / Math.max(1, numSteps - 1);

  // Plot trajectories for Pan, Lift, Elbow, Gripper
  [0, 1, 2, 5].forEach((jointIdx) => {
    ctx.strokeStyle = colors[jointIdx];
    ctx.lineWidth = 2;
    ctx.beginPath();

    for (let step = 0; step < numSteps; step++) {
      const val = predChunk[step][jointIdx]; // degrees, e.g. -110 to +110
      // Map [-120, 120] to canvas height [h, 0]
      const y = h - ((val + 120) / 240) * h;
      const x = step * stepWidth;

      if (step === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  });

  // Draw Legend
  ctx.font = '10px monospace';
  ctx.fillStyle = '#f43f5e';
  ctx.fillText('● Lift', 10, 16);
  ctx.fillStyle = '#a855f7';
  ctx.fillText('● Elbow', 60, 16);
  ctx.fillStyle = '#38bdf8';
  ctx.fillText('● Pan', 115, 16);
  ctx.fillStyle = '#4ade80';
  ctx.fillText('● Gripper', 160, 16);
}

async function pollAIEvalStatus() {
  try {
    const res = await api('ai/eval/status');
    if (!res) return;

    const startBtn = document.getElementById('btn-start-eval');
    const stopBtn = document.getElementById('btn-stop-eval');
    const statusText = document.getElementById('ai-eval-status-text');
    const fpsBadge = document.getElementById('ai-eval-fps-badge');

    if (startBtn) startBtn.disabled = res.is_evaluating;
    if (stopBtn) stopBtn.disabled = !res.is_evaluating;

    if (statusText) {
      if (res.is_evaluating) {
        statusText.innerHTML = `<span style="color:#10b981; font-weight:700;">🟢 Autonomous policy running — picking up objects autonomously!</span>`;
      } else {
        statusText.textContent = 'Autonomous execution idle. Select a model and click Run.';
      }
    }

    if (fpsBadge) {
      fpsBadge.textContent = res.is_evaluating ? `${res.fps || 30.0} FPS` : '-- FPS';
    }

    // Update Live Joint Telemetry Card
    if (res.current_action) {
      const act = res.current_action;
      const setVal = (id, k) => {
        const el = document.getElementById(id);
        if (el && act[k] !== undefined) el.textContent = `${Number(act[k]).toFixed(1)}°`;
      };
      setVal('ai-eval-pan', 'shoulder_pan.pos');
      setVal('ai-eval-lift', 'shoulder_lift.pos');
      setVal('ai-eval-elbow', 'elbow_flex.pos');
      setVal('ai-eval-wristf', 'wrist_flex.pos');
      setVal('ai-eval-wristr', 'wrist_roll.pos');
      setVal('ai-eval-gripper', 'gripper.pos');
    }

    // Render Live Trajectory Waveform Canvas
    if (res.predicted_trajectory) {
      renderTrajectoryWaveform(res.predicted_trajectory);
    }

    if (res.is_evaluating) {
      setTimeout(pollAIEvalStatus, 200);
    }
  } catch (_) {}
}



