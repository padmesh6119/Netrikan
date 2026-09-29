# Netrikan — what to do next (Pranav, Vishruth)

Written 2026-09-29. Repo branch for the new experiment: `exp/dapt-identity-ab`.
**Assumed split:** Pranav = data + model, Vishruth = deck + demo. Swap if that's wrong.

---

## Where we actually are (read this first, 1 minute)

- **The model works, but it is overfitted to CIC-IDS-2018.** It scores 0.90 onset AUC on held-out CIC windows, 0.64 on held-out days, and 0.70 on DAPT (a network it never saw). Transfer to DAPT is best after **1 epoch** and gets worse with more training (`models/transfer_vs_epoch.json`).
- **Knob-turning is done.** Attention, balancing, state weight, epochs, ensembles and horizon sweeps all move the score by about ±0.005. Don't retrain CIC again.
- **New lead: per-host windows.** On DAPT, windows built from one host's flows forecast much better than whole-network windows: onset AUC **0.84 vs 0.65** (`models/dapt_identity_ab.json`).
  - **But this rests on ONE held-out capture with 188 stage changes.** It is a hint, not a result.
  - **Do not quote it anywhere yet.**

---

## Pranav — data + model

### Do

1. **Finish the UWF Zeek download.** This is the dataset that can confirm the per-host result at scale.
2. **Write a loader for UWF.** Map Zeek `conn.log` fields onto as many of our 24 features as honestly map (duration, bytes and packets per direction, and so on).
   - Features with no equivalent get dropped, not faked.
   - Keep `Src IP`, `ts` and the ATT&CK label per row.
   - Output a DataFrame with the same columns `src/dapt.py:load_all()` returns (`capture`, `Src IP`, `ts`, `stage_id`, plus the features), so `bench/dapt_identity_ab.py` runs on it with no other changes.
3. **Run the per-host vs whole-network comparison on UWF:**
   - Group hosts across the **whole capture period**, not per file.
   - Hold out **hosts** (5 groups by IP), not captures.
   - Score both versions on the **same target**: "does this host change stage in the next k steps".
4. **Keep DAPT as the untouched test set.** Train on UWF, test on DAPT. That keeps our "attack behaviour transfers across networks" claim alive.
5. **Stop training early.** Pick the checkpoint on a held-out day or held-out network, not on in-dataset score. 1–3 epochs.
6. **Every result goes in a JSON in `models/`**, produced by a script in `bench/`, committed together.

### Don't

- ❌ Don't retrain on CIC-IDS-2018 with new settings. It's plateaued.
- ❌ Don't train on DAPT. It's our only unseen-network test. Once it's in training, we lose that evidence.
- ❌ Don't try `pipeline_identity.py` on raw CIC PCAPs. That's 450 GB, and this disk has 2 GB free.
- ❌ Don't invent Zeek-to-CIC feature mappings to reach 24 features. Fewer honest features beat 24 fake ones.
- ❌ Don't report `stage_macro_f1_DO_NOT_REPORT`, all-window "beats persistence", or any single-fold number as a headline.
- ❌ Never write a number into a JSON or doc by hand. If a script didn't produce it, it doesn't exist.

**Done when:** `models/uwf_identity_ab.json` exists with **≥5 scored folds**, and per-host either wins on most of them or doesn't. Both outcomes are useful. If it doesn't win, we drop the per-host claim and stay with "network-level early warning".

---

## Vishruth — deck + demo

### Do

1. **Build the 6-slide idea deck from `PPT_CONTENT.md`**, in the official template `SIH2026-IDEA-Presentation-Format.pptx`.
   - Keep the template's section headings exactly as they are.
   - Delete slide 7 (the instructions).
   - Export as **PDF**.
2. **Use only the numbers in `PPT_CONTENT.md`.** Its §7 table says which file each one comes from.
3. **Slide 2 opens with the clock: "13.5 min"**, the median warning before an attack starts.
4. **Slide 4 carries the "we tried to break our own model" strip**: held-out days, an unseen network, shuffled time order, 4 seeds.
5. **Dashboard screenshot:** run `streamlit run src/app.py`, pick the "Intrusion" scenario, and capture the risk curve with the forecast marker.
6. **60-second demo video** with a QR code on slide 3. Replay a capture and show the forecast firing *before* the attack stage arrives. Then click "Simulate interventions" to show the risk drop.
7. **Fill in the Team ID, the team name and the exact problem-statement title from the portal.**

### Don't

- ❌ Don't add per-host / "tracks the individual attacker" as a feature. It stays **roadmap** until Pranav's UWF result lands.
- ❌ Don't show SPRT, "Time Bought", a Suricata comparison or D3FEND as built. They're roadmap.
- ❌ Don't write "99% accuracy", "beats persistence" without "on transition windows", or any number not in `PPT_CONTENT.md`.
- ❌ Don't name competitor teams on the slides.
- ❌ Don't go past 6 slides, and don't use paragraphs. Points, diagrams, big numbers.
- ❌ Don't change the template's headings or layout frame. SIH rejects modified templates.

**Done when:** a 6-page PDF has been reviewed against the "Do NOT put on any slide" list in `PPT_CONTENT.md` §5.

---

## Both

- If a number in the deck and a JSON in `models/` disagree, **the JSON wins**. Fix the deck.
- Anything uncertain: ask before you claim it. A judge catching one inflated number costs more than the number was worth.
