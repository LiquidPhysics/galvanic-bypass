// Export each example case's 8-cell module telemetry (6 h, REON off so the readings are the raw
// symptoms) in celldx's common schema.
import { writeFileSync } from 'node:fs';
import { CASES, CASE_KEYS, initCase, advance } from '/home/user/heat-atlas/src/model/cases';
const out = process.argv[2];
for (const k of CASE_KEYS) {
  const spec = CASES[k];
  const s = initCase(spec);
  const rows = ['t,cell_id,pack_id,string_id,voltage_v,current_a,temp_cell_c,temp_pack_c,soc,soh,resistance_ohm'];
  for (let m = 0; m <= 360; m++) {
    for (const c of s.cells) rows.push([Math.round(s.t), c.label, `case-${k}`, 'A', c.V.toFixed(4), c.I.toFixed(3), c.T.toFixed(3), s.ambient.toFixed(3), c.soc.toFixed(4), c.soh.toFixed(4), (c.R_mOhm / 1000).toFixed(6)].join(','));
    advance(spec, s, 1, false, () => 'healthy');
  }
  writeFileSync(`${out}/case_${k}.csv`, rows.join('\n'));
}
console.log('ok');
