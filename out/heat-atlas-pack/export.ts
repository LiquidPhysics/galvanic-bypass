// Export the Heat Atlas simulated pack (the one every page shows) for celldx:
//  - pack_firebreak.csv: minute telemetry in celldx's common schema (what the pages display)
//  - frames_baseline.jsonl: fixed-string frames (what a plain REON pack streams) for the thermal controller
import { writeFileSync } from 'node:fs';
import { simulate } from '/home/user/heat-atlas/src/model/sim';
import { DEFAULT_PARAMS } from '/home/user/heat-atlas/src/model/constants';
import { cv, pv } from '/home/user/heat-atlas/src/model/tape';

const out = process.argv[2];
const r = simulate({ params: DEFAULT_PARAMS, scenario: 'hot-microgrid', ageYears: 2 });
const tp = r.tapes.firebreak;
const rows = ['t,cell_id,pack_id,string_id,voltage_v,current_a,temp_cell_c,temp_pack_c,soc,soh,resistance_ohm'];
for (let s = 0; s < tp.steps; s++) {
  const t = pv(tp, s, 't'), amb = pv(tp, s, 'ambient');
  for (let i = 0; i < tp.n; i++) rows.push([t, tp.ids[i], 'S-12', 'string0', cv(tp, s, i, 'V').toFixed(4), cv(tp, s, i, 'I').toFixed(3), cv(tp, s, i, 'T').toFixed(3), amb.toFixed(3), cv(tp, s, i, 'soc').toFixed(4), cv(tp, s, i, 'soh').toFixed(4), (cv(tp, s, i, 'R') / 1000).toFixed(6)].join(','));
}
writeFileSync(`${out}/pack_firebreak.csv`, rows.join('\n'));
const b = r.tapes.baseline;
const fr: string[] = [];
for (let s = 0; s < b.steps; s += 5) {
  fr.push(JSON.stringify({ t: pv(b, s, 't'), ambient_C: pv(b, s, 'ambient'), request_W: pv(b, s, 'request_W'),
    cells: b.ids.map((id, i) => ({ id, V: cv(b, s, i, 'V'), I: cv(b, s, i, 'I'), T: cv(b, s, i, 'T'), soc: cv(b, s, i, 'soc'), soh: cv(b, s, i, 'soh'), R_mOhm: cv(b, s, i, 'R') })) }));
}
writeFileSync(`${out}/frames_baseline.jsonl`, fr.join('\n') + '\n');
console.log('steps', tp.steps, 'dt', tp.dt, 'cells', tp.n, 'ids', tp.ids.slice(0, 3), 'frames', fr.length);
console.log(JSON.stringify(r.layout).slice(0, 200));
