"""Record a ~90-second demo video of the running web app (Phase 8 extra).

Drives the installed Microsoft Edge with Playwright (no browser download), records the page at
1920x1080 and converts the recording to MP4 (H.264) with ffmpeg from imageio-ffmpeg. Captions are
injected into the recorded page only; the app itself is unchanged.

Needs the demo running (start_demo.ps1) and, OUTSIDE the project environment:
    python -m venv videnv && videnv\\Scripts\\pip install playwright imageio-ffmpeg
Usage:
    videnv\\Scripts\\python.exe scripts\\record_demo_video.py --geo-job <job id of a processed Pittsburgh GeoTIFF>
"""
from __future__ import annotations

import argparse
import time
import shutil
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
APP = "http://127.0.0.1:5173"
W, H = 1920, 1080

CAPTION_JS = """
window.__cap = (title, sub, where) => {
  let el = document.getElementById('__dwcap');
  if (!el) {
    el = document.createElement('div'); el.id = '__dwcap';
    el.style.cssText = 'position:fixed;left:50%;transform:translateX(-50%);z-index:2147483647;' +
      'background:rgba(2,6,23,.9);border:1px solid rgba(56,189,248,.55);border-radius:14px;' +
      'padding:14px 26px;font:600 26px system-ui,Segoe UI,sans-serif;color:#f1f5f9;text-align:center;' +
      'pointer-events:none;box-shadow:0 10px 40px rgba(0,0,0,.55);max-width:62%;transition:opacity .25s';
    document.body.appendChild(el);
  }
  el.style.top = where === 'top' ? '64px' : 'auto';
  el.style.bottom = where === 'top' ? 'auto' : '72px';
  el.innerHTML = title + (sub ? '<div style="font:400 17px system-ui,Segoe UI,sans-serif;color:#94a3b8;margin-top:6px">' + sub + '</div>' : '');
  el.style.opacity = title ? '1' : '0';
};
"""


T0 = [0.0]


def cap(page, title, sub="", where="bottom"):
    print(f"{time.monotonic() - T0[0]:6.1f}s  {title}", flush=True)
    page.evaluate("([t, s, w]) => window.__cap(t, s, w)", [title, sub, where])


def box(page, selector):
    b = page.locator(selector).first.bounding_box()
    return b["x"], b["y"], b["width"], b["height"]


def drag(page, x0, y0, x1, y1, steps=90, button="left"):
    page.mouse.move(x0, y0)
    page.mouse.down(button=button)
    page.mouse.move(x1, y1, steps=steps)
    page.mouse.up(button=button)


def wait_loaded(page):
    """Wait until the Explorer's 'Loading terrain' indicator is gone (texture and mesh ready)."""
    page.wait_for_timeout(300)
    page.wait_for_function("() => !document.body.innerText.includes('Loading terrain')", timeout=60000)


def glide(page, pts, steps=40):
    page.mouse.move(*pts[0])
    for p in pts[1:]:
        page.mouse.move(*p, steps=steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geo-job", required=True)
    ap.add_argument("--png", default=str(ROOT / "sample_data" / "gamus_val" / "DC_48_31_rgb.png"))
    ap.add_argument("--lidar", default=str(ROOT / "sample_data" / "geo" / "pittsburgh_lidar_dsm.tif"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "media" / "depthwizard_demo.mp4"))
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="dwvid_"))

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=False,
                                    args=["--window-size=1936,1160", "--force-device-scale-factor=1",
                                          "--ignore-gpu-blocklist"])
        ctx = browser.new_context(viewport={"width": W, "height": H}, device_scale_factor=1,
                                  record_video_dir=str(tmp), record_video_size={"width": W, "height": H})
        ctx.add_init_script(CAPTION_JS)
        page = ctx.new_page()
        T0[0] = time.monotonic()

        # ---- 0: title
        page.goto(APP); page.wait_for_selector("text=Input Image")
        cap(page, "DepthWizard", "Height in metres from a single aerial image · SIH 2026 · PS 26175 (ISRO)")
        page.wait_for_timeout(3000)

        # ---- 1: upload a single top-down image
        cap(page, "1 · Upload one top-down image", "GAMUS validation tile (Washington DC), 0.33 m per pixel")
        page.locator('input[type=file][accept*="image/jpeg"]').set_input_files(a.png)
        page.wait_for_timeout(600)
        page.locator("input[type=number]").fill("0.33")
        page.wait_for_timeout(500)
        page.get_by_role("button", name="Generate Terrain").click()
        page.get_by_role("button", name="Enter 3D Explorer ⛶").wait_for(timeout=60000)
        page.wait_for_timeout(800)

        # ---- 2: result on the dashboard
        cap(page, "2 · Height above ground, in metres", "Depth Anything V2 Small fine-tuned on LiDAR heights · ~2 s on the GPU")
        x, y, w, h = box(page, "canvas")
        glide(page, [(x + w * .35, y + h * .55), (x + w * .5, y + h * .45), (x + w * .62, y + h * .6)], steps=15)
        page.wait_for_timeout(500)
        drag(page, x + w * .3, y + h * .7, x + w * .55, y + h * .66, steps=50)
        page.wait_for_timeout(800)

        # ---- 3: Explorer, layers, profile
        cap(page, "3 · 3D Explorer at true scale (1 m = 1 m)", "layers: photo, height, slope · profile · measure")
        page.get_by_role("button", name="Enter 3D Explorer ⛶").click()
        page.wait_for_selector("#dw-explorer-canvas canvas"); wait_loaded(page); page.wait_for_timeout(800)
        ex, ey, ew, eh = box(page, "#dw-explorer-canvas canvas")
        drag(page, ex + ew * .30, ey + eh * .6, ex + ew * .45, ey + eh * .57, steps=50)
        page.wait_for_timeout(400)
        for label in ("Height above ground", "Slope (degrees)", "RGB image"):
            page.locator("label", has_text=label).click(); page.wait_for_timeout(1100)
        page.locator('button[title^="Height profile"]').click(); page.wait_for_timeout(500)
        page.mouse.click(ex + ew * .30, ey + eh * .55); page.wait_for_timeout(500)
        page.mouse.click(ex + ew * .70, ey + eh * .58)
        cap(page, "Height profile across the city block", "", where="top")
        page.wait_for_timeout(2500)
        page.locator('button[title^="Height profile"]').click(); page.wait_for_timeout(300)

        # ---- 4: fly and walk
        cap(page, "4 · Fly and walk through it", "every step is checked against the surface · fly ≥ 2 m up · walk at eye height 1.7 m")
        page.keyboard.press("Digit2"); page.wait_for_timeout(600)
        page.mouse.click(ex + ew * .5, ey + eh * .5); page.wait_for_timeout(400)
        # fly in from the overview position (outside the image) towards the centre: Shift = 4x speed
        page.keyboard.down("ShiftLeft"); page.keyboard.down("KeyW"); page.wait_for_timeout(3200)
        page.keyboard.up("ShiftLeft"); page.wait_for_timeout(1200); page.keyboard.up("KeyW")
        page.wait_for_timeout(400)
        page.keyboard.press("Digit3"); page.wait_for_timeout(600)         # walk: eye height 1.7 m
        page.keyboard.down("KeyW"); page.wait_for_timeout(2000); page.keyboard.up("KeyW")
        page.wait_for_timeout(400)

        # ---- 5: GeoTIFF -> absolute elevation
        page.goto(f"{APP}/?job={a.geo_job}")
        page.get_by_role("button", name="Enter 3D Explorer ⛶").wait_for(timeout=60000)
        cap(page, "5 · GeoTIFF in → absolute elevation out",
            "CRS read from the file · Copernicus GLO-30 ground + GCP correction · metres above EGM2008 · GeoTIFF export")
        page.wait_for_timeout(2000)
        page.evaluate("window.scrollTo(0, 0)")
        page.get_by_role("button", name="Enter 3D Explorer ⛶").click()
        page.wait_for_selector("#dw-explorer-canvas canvas"); wait_loaded(page); page.wait_for_timeout(600)
        page.locator("label", has_text="Elevation (EGM2008)").click(); page.wait_for_timeout(500)
        ex, ey, ew, eh = box(page, "#dw-explorer-canvas canvas")
        drag(page, ex + ew * .35, ey + eh * .62, ex + ew * .5, ey + eh * .59, steps=50)
        cap(page, "Elevation, height above ground, easting/northing and lat/lon under the cursor", "", where="top")
        glide(page, [(ex + ew * .4, ey + eh * .5), (ex + ew * .55, ey + eh * .45), (ex + ew * .6, ey + eh * .55)], steps=15)
        page.wait_for_timeout(900)

        # ---- 6: validation against LiDAR
        page.locator('input[accept=".npy,.tif,.tiff"]').set_input_files(a.lidar)
        page.wait_for_selector("text=full-resolution error map", timeout=60000)
        cap(page, "6 · Checked against USGS LiDAR", "reference datum NAVD88 → EGM2008 converted automatically · error map in 3D")
        drag(page, ex + ew * .55, ey + eh * .6, ex + ew * .42, ey + eh * .6, steps=50)
        page.wait_for_timeout(1800)

        # ---- end
        cap(page, "DepthWizard", "one image → metres → 3D · validated against LiDAR · runs offline on a laptop GPU")
        page.wait_for_timeout(2800)
        video = page.video.path()
        ctx.close(); browser.close()

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(video), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", "-preset", "slow", "-r", "30", "-movflags", "+faststart", str(out)], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
