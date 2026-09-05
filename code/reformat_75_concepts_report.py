import re
from pathlib import Path

src = Path(r'D:/Coding/BrainAI/ResNet50_Final_AllUnits_2026-08-24/results_75_concepts_units_pca32.md')
raw = Path(r'D:/Coding/BrainAI/ResNet50_Final_AllUnits_2026-08-24/results_75_concepts_units_pca32_raw.txt')
text = raw.read_text(encoding='utf-8', errors='replace')
layer = None
rows = {}
for line in text.splitlines():
    m = re.match(r'^\[(res[2345])\]$', line.strip())
    if m:
        layer = m.group(1)
        continue
    m = re.match(r'^(\d+) (\d+)-(\d+) r=([+-]?\d*\.\d+): (.+)$', line.strip())
    if m and layer:
        unit, start, end, r, top = m.groups()
        rows[(int(unit), layer, int(start))] = (int(end), r, top)

roi_units = {
    'V1': (19783, 18219), 'V1/V2': (16431, 16277), 'V2': (16010, 17395),
    'V4': (19254, 18896), 'CLC': (11947, 5791),
}
windows = (70, 110, 150)
layers = ('res2', 'res3', 'res4', 'res5')
out = [
    '# PCA32 75个高质量概念轴：逐 Unit 结果', '',
    '筛选：正例数 >1000，且四层最低重复轴余弦 >0.9。',
    '每个 unit 下按层列出 70–79、110–119、150–159 ms 三个窗口；数值为五折 OOF Pearson r 和 signed cosine top5。', '',
]
for roi, units in roi_units.items():
    out += [f'## {roi}', '']
    for j, unit in enumerate(units, 1):
        label = f'{roi}-{j}' if roi in ('V1', 'V1/V2', 'V2', 'V4') else f'{roi}{j}'
        out += [f'### {label}', f'原始 unit：`{unit}`', '']
        for l in layers:
            out += [f'#### {l}', '', '| 时间窗 | OOF r | Top-5 概念轴（signed cosine） |', '|---|---:|---|']
            for start in windows:
                value = rows.get((unit, l, start))
                if value is None:
                    out.append(f'| {start}–{start+9} ms | NA | NA |')
                else:
                    end, r, top = value
                    out.append(f'| {start}–{end} ms | {r} | {top.replace(" | ", "; ")} |')
            out.append('')
        out.append('')
src.write_text('\n'.join(out), encoding='utf-8')
print(src)
print('units', sum(len(x) for x in roi_units.values()), 'rows', len(rows))
