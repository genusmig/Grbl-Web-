#!/usr/bin/env node
/**
 * Helper: convert text+local font file to a single SVG path.
 * Tries google-font-to-svg-path; falls back to opentype.js for local fonts.
 * Usage: node font_to_svg_path.js --font <path> --text <text> [--fontSize 72] [--letterSpacing 0]
 * Outputs JSON: { path: string, width: number, height: number }
 */
const fs = require('fs');
const path = require('path');

function arg(name, def) {
  const i = process.argv.indexOf(`--${name}`);
  if (i >= 0 && i + 1 < process.argv.length) return process.argv[i + 1];
  return def;
}

async function tryGoogleFontToSvgPath(fontFile, text, fontSize, letterSpacing) {
  try {
    const gfts = require('google-font-to-svg-path');
    const toPath = gfts.default || gfts;
    const res = await toPath({
      text,
      font: `local:${path.resolve(fontFile)}`,
      fontSize,
      letterSpacing,
      normalize: true,
    });
    return { path: res.path || '', width: Number(res.width || 0), height: Number(res.height || 0) };
  } catch (e) {
    return null;
  }
}

async function viaOpenType(fontFile, text, fontSize, letterSpacing) {
  const opentype = require('opentype.js');
  const font = await new Promise((resolve, reject) => {
    opentype.load(fontFile, (err, f) => (err ? reject(err) : resolve(f)));
  });
  const scale = 1;
  const baseline = 0; // opentype path is relative to baseline
  let x = 0;
  let y = baseline;
  let d = '';
  const commandsToPath = (cmds) => {
    let s = '';
    for (const c of cmds) {
      if (c.type === 'M') s += `M${c.x} ${-c.y}`;
      else if (c.type === 'L') s += `L${c.x} ${-c.y}`;
      else if (c.type === 'C') s += `C${c.x1} ${-c.y1} ${c.x2} ${-c.y2} ${c.x} ${-c.y}`;
      else if (c.type === 'Q') s += `Q${c.x1} ${-c.y1} ${c.x} ${-c.y}`;
      else if (c.type === 'Z') s += 'Z';
    }
    return s;
  };
  for (const ch of text) {
    const glyph = font.charToGlyph(ch);
    const gPath = glyph.getPath(x, y, fontSize, { kerning: true });
    d += commandsToPath(gPath.commands);
    const adv = glyph.advanceWidth * (fontSize / font.unitsPerEm);
    x += adv + letterSpacing;
  }
  // width as accumulated x, approximate height from font metrics
  const width = x;
  const ascent = font.ascender * (fontSize / font.unitsPerEm);
  const descent = font.descender * (fontSize / font.unitsPerEm);
  const height = ascent - descent;
  return { path: d, width, height };
}

async function viaOpenTypeMulti(fontFile, text, fontSize, letterSpacing, lineSpacingFactor, align = 'left') {
  const opentype = require('opentype.js');
  const font = await new Promise((resolve, reject) => {
    opentype.load(fontFile, (err, f) => (err ? reject(err) : resolve(f)));
  });
  const ascent = font.ascender * (fontSize / font.unitsPerEm);
  const descent = font.descender * (fontSize / font.unitsPerEm);
  const baseLineHeight = ascent - descent;
  const lineHeight = baseLineHeight * (1 + (isFinite(lineSpacingFactor) ? lineSpacingFactor : 0.2));

  const commandsToPath = (cmds) => {
    let s = '';
    for (const c of cmds) {
      if (c.type === 'M') s += `M${c.x} ${-c.y}`;
      else if (c.type === 'L') s += `L${c.x} ${-c.y}`;
      else if (c.type === 'C') s += `C${c.x1} ${-c.y1} ${c.x2} ${-c.y2} ${c.x} ${-c.y}`;
      else if (c.type === 'Q') s += `Q${c.x1} ${-c.y1} ${c.x} ${-c.y}`;
      else if (c.type === 'Z') s += 'Z';
    }
    return s;
  };

  const lines = String(text).split(/\r?\n/);
  // First pass: measure each line width
  const widths = lines.map((line) => {
    let x = 0;
    for (const ch of line) {
      const glyph = font.charToGlyph(ch);
      const adv = glyph.advanceWidth * (fontSize / font.unitsPerEm);
      x += adv + letterSpacing;
    }
    return x;
  });
  const maxWidth = widths.reduce((a, b) => Math.max(a, b), 0);

  // Second pass: build path with alignment offset per line
  let d = '';
  lines.forEach((line, idx) => {
    let x = 0;
    const w = widths[idx];
    let xOffset = 0;
    const a = String(align || 'left').toLowerCase();
    if (a === 'right') xOffset = maxWidth - w;
    else if (a === 'center' || a === 'centre') xOffset = (maxWidth - w) / 2;
    const y = idx * lineHeight;
    for (const ch of line) {
      const glyph = font.charToGlyph(ch);
      const gPath = glyph.getPath(x + xOffset, y, fontSize, { kerning: true });
      d += commandsToPath(gPath.commands);
      const adv = glyph.advanceWidth * (fontSize / font.unitsPerEm);
      x += adv + letterSpacing;
    }
  });

  const totalHeight = Math.max(lineHeight * lines.length, baseLineHeight);
  return { path: d, width: maxWidth, height: totalHeight };
}

(async () => {
  try {
    const fontFile = arg('font');
    const text = arg('text', '') || '';
    const fontSize = parseFloat(arg('fontSize', '72'));
    const letterSpacing = parseFloat(arg('letterSpacing', '0'));
  const lineSpacing = parseFloat(arg('lineSpacing', '0.2'));
  const align = String(arg('align', 'left') || 'left').toLowerCase();
    if (!fontFile) throw new Error('Missing --font');

    let result = null;
    const multi = text.includes('\n') || text.includes('\r');
    if (multi) {
      // Multi-line: OpenType with alignment
      result = await viaOpenTypeMulti(fontFile, text, fontSize, letterSpacing, lineSpacing, align);
    } else {
      // Single-line: if alignment requested, use OpenType so we can offset reliably
      if (align !== 'left') {
        result = await viaOpenTypeMulti(fontFile, text, fontSize, letterSpacing, lineSpacing, align);
      } else {
        // Try google font path first (fast), then fallback
        result = await tryGoogleFontToSvgPath(fontFile, text, fontSize, letterSpacing);
        if (!result) {
          try {
            result = await viaOpenType(fontFile, text, fontSize, letterSpacing);
          } catch (e) {
            console.error('Please install dependencies: npm i google-font-to-svg-path opentype.js');
            throw e;
          }
        }
      }
    }
    process.stdout.write(JSON.stringify(result));
  } catch (err) {
    console.error(String((err && err.stack) || err));
    process.exit(1);
  }
})();
