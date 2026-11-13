# Commercial-use plan: English–Sorani data and model selection

**Status: partial, synthetic-only implementation.** You confirmed that both the app and the trained model are intended for commercial use. Offline [paired-line](text-pair-staging.md) and [plain-segment TMX](tmx-pair-staging.md) staging CLIs and a [pair-overlap review audit](overlap-audit.md) have been implemented and tested with synthetic examples only. They do not produce a trainer-ready dataset or validate legal rights, language, translations, or independent works. SMOL JSONL, text-only training integration, an independent commercial subtitle evaluation, and a commercial-compatible model remain future work. No real corpus has been staged or imported, and no model has been trained, activated, or deployed as part of this plan. The Windows job has not been changed and must not be stopped or restarted without your agreement. This plan replaces the earlier non-commercial InterdialectCorpus pilot.

## Recommendation in plain language

1. **Do not use the current NLLB checkpoint or InterdialectCorpus for this commercial project** unless the necessary separate commercial permissions are obtained. Fine-tuning does not remove the base model's restrictions.
2. **Find or commission authorized subtitle data**, including separate, genuinely independent films or works for validation and final testing. A file being publicly available does not by itself grant training rights.
3. **Investigate Mozilla's English–Central Kurdish translation memory** as a general software-localization lead, not subtitles. The repository has an MPL-2.0 license, but the underlying text, translation rights, obligations, language, and quality still need review. The new TMX importer can be exercised with synthetic data in the meantime.
4. **Investigate Google SMOL's English–Sorani sentence subset** as another small, general-domain lead, subject to review of its CC BY 4.0 terms and rights in its original web-sourced English text. It is not subtitle dialogue and is not cleared for this project yet.
5. **Investigate the smaller Helsinki Marian models first**, one per translation direction, because their weight files are much smaller than MADLAD-400's. Investigate MADLAD-400 if the hardware and disk budget permit. These are candidates, not drop-in replacements or quality guarantees.
6. Only after rights, user agreement, integration tests, hardware checks, and independent bilingual subtitle evaluation should a commercial model be considered for deployment.

This document is an engineering plan, not legal advice. An Apache or Creative Commons label is useful evidence, but does not by itself establish all underlying rights or complete the required attribution and other obligations.

## What is blocked for commercial use

- **Current NLLB-200 distilled 600M base:** its model card states CC BY-NC 4.0 and describes it as a research model not released for production deployment [3](https://huggingface.co/facebook/nllb-200-distilled-600M/blob/main/README.md). Do not use it, or a LoRA/merged derivative based on it, commercially without appropriate separate rights. Also review whether any outputs or feedback derived from it may be used in this commercial training pipeline; do not assume those rights.
- **KurdishBLARK InterdialectCorpus:** the repository states CC BY-NC-SA 4.0, reports 650 English–Sorani pairs, and asks users to contact the contributors about commercial use [1](https://github.com/KurdishBLARK/InterdialectCorpus). Do not import it for this commercial project unless the necessary permissions, including underlying news-text rights, are established. It is news text, not film dialogue.
- **Awta and other unclear sources:** the Awta repository files previously inspected did not establish a clear dataset license. Do not use it until the intended-use permissions and upstream rights are resolved. Repository or source-code licenses do not automatically grant rights to every contained text.

The Windows translation job's current status is unknown. It was configured to use NLLB, so do not treat its output as commercially cleared without a rights review. This plan does **not** order a cancellation, restart, or change to the running system. Check its status without changing it in the signed-in app's Jobs page; a stale progress display alone does not prove it has stopped. Specific private job identifiers are intentionally omitted from this public document.

## Candidate data: what to acquire or investigate

### 1. Authorized subtitle-domain material — highest priority

Obtain English and Sorani subtitle pairs for works you own or have permission to use, or commission original dialogue and translations. Document rights for **both** languages, the original work, the translator's contribution, training, validation, commercial operation, and any intended distribution. Include any relevant term, territory, retention, privacy, and third-party-processing conditions. Having permission to watch a film, download a subtitle, or translate it for personal viewing is not automatically permission to train a commercial model.

Keep separate, genuinely independent works for model development and final testing. If subtitle examples are used for training too, use different works for training, validation, and test. If a general-text source is used for training instead, still obtain separate independent subtitle works for validation and test. More works and varied dialogue are needed for a reliable assessment; a minimum number of files does not ensure statistical reliability.

If no suitable existing material is available, one option is to commission original short scripts and human Sorani translations with real subtitle timings and explicit written commercial AI-use rights. This avoids assuming that subtitles from commercial films are freely usable, although a small commissioned set will not represent all film genres.

**A small timed-subtitle lead, not a cleared corpus:** the Wikimedia Commons video [What is Wikipedia Zero?](https://commons.wikimedia.org/wiki/File:What_is_Wikipedia_Zero%3F.webm) is marked CC BY-SA 3.0, and its separately hosted [English](https://commons.wikimedia.org/w/index.php?title=TimedText:What_is_Wikipedia_Zero%3F.webm.en.srt&oldid=150557152) and [Central Kurdish](https://commons.wikimedia.org/w/index.php?title=TimedText:What_is_Wikipedia_Zero%3F.webm.ckb.srt&oldid=427072615) tracks contain 24 cues each with matching time intervals. The English [revision history](https://commons.wikimedia.org/w/api.php?action=query&prop=revisions&titles=TimedText%3AWhat_is_Wikipedia_Zero%3F.webm.en.srt&rvprop=ids%7Ctimestamp%7Cuser%7Ccomment&rvlimit=50&format=json&formatversion=2) identifies `VGrigas (WMF)` as editor; the Kurdish [history](https://commons.wikimedia.org/w/api.php?action=query&prop=revisions&titles=TimedText%3AWhat_is_Wikipedia_Zero%3F.webm.ckb.srt&rvprop=ids%7Ctimestamp%7Cuser%7Ccomment&rvlimit=50&format=json&formatversion=2) identifies `Aram`, who described completing the translation. This is useful provenance evidence, not independent proof of rights or accuracy. Review the terms applicable to the actual subtitle revisions, the editor's authority, attribution, share-alike and any trademark obligations; the video's license alone does not settle the translation's rights. A bilingual reviewer must check meaning and cue alignment. It is one short promotional video, not film dialogue, and far too small by itself for model training or a representative independent test. It was inspected, not imported.

### 2. Mozilla Pontoon — general localization-text lead, not subtitles

[Mozilla's translation-memory repository](https://github.com/mozilla-l10n/mt-training-data) describes its purpose as language data for machine-translation localization and has an [MPL-2.0 license](https://github.com/mozilla-l10n/mt-training-data/blob/cda8e0ccfa43c2ce14b6a24923792036be2f1b61/LICENSE), which permits commercial use subject to its terms. Its [Central Kurdish directory at revision `cda8e0ccfa43c2ce14b6a24923792036be2f1b61`](https://github.com/mozilla-l10n/mt-training-data/tree/cda8e0ccfa43c2ce14b6a24923792036be2f1b61/ckb) contains `ckb_pontoon.tmx`, an aggregated Pontoon translation-memory export. This is a promising source for investigation, but a repository license is not a blanket clearance of all underlying source strings, translations, project-specific provenance, or model/distribution obligations. Review the relevant rights, MPL notices and conditions, possible third-party contributions, and intended handling of models before commercial use. The language label and `srclang` declarations do not independently verify dialect or translation quality.

The exact file at that revision was **inspected in memory, not saved or staged**. It contained 8,999 two-variant translation units tagged `en-US` and `ckb`; 2,880 were repeated exact English/Sorani text pairs after trimming, leaving 6,119 distinct exact text pairs by that simple count. There were also 111 repeated `tuid` occurrences beyond the first. Some texts are multiline or contain markup/variables, and project IDs do not establish independent works. These are localization strings, not a subtitle corpus, and they may overlap other sources or model pretraining. The file was 2,597,833 bytes with SHA-256 `677096583c597da191f2c3d8a6f5cd138e9f6b984dde4b5717b2447dfa80f555`. Those are observations of one revision, not a bilingual, rights, or alignment audit. The [new TMX staging tool](tmx-pair-staging.md) was tested only with synthetic fixtures; its ability to process the actual file has not been exercised.

### 3. SMOL / SmolSent — general-text training lead, not an approval

The [Google SMOL dataset card](https://huggingface.co/datasets/google/smol) states **CC BY 4.0**, lists **863** English–Sorani `SmolSent` sentence pairs, and credits Sorani volunteer work. It separately lists **3,986** `Gatitos` lexicon entries; those are predominantly words or short phrases, **not another 3,986 subtitle sentences** [1](https://huggingface.co/datasets/google/smol). The paper says SmolSent's English sentences were selected from Common Crawl [1](https://arxiv.org/html/2502.12301v2). Review the collection license, required attribution, original-source rights and any missing provenance before authorizing commercial training or redistribution. If the rights cannot be resolved, exclude the affected material rather than assume that the dataset label clears it.

The publisher identifies `smolsent/en_ckb.jsonl` as the relevant subset. An identifiable upstream revision for investigation is `d316fa4aa5a5973acea5c7878a48e73c9fcf8016`:

<https://huggingface.co/datasets/google/smol/blob/d316fa4aa5a5973acea5c7878a48e73c9fcf8016/smolsent/en_ckb.jsonl>

The dataset card describes records with `id`, `sl`, `tl`, `src`, `trg`, and `is_src_orig`; source and target are strings [2](https://huggingface.co/datasets/google/smol/blob/main/README.md). The beginning of this file was viewed, but it has **not** been fully audited, imported, or saved as project training data. The 863 count is the publisher's reported count, not our audit. Pin the chosen revision and hash the actual file if a future import is approved. Treat these as isolated, general-domain pairs with no established film timing or consecutive dialogue context. This is a small, out-of-domain resource: fine-tuning on it might help, make no difference, or worsen subtitle translation. Preserve original-direction metadata when reversing pairs for a separately configured Sorani→English experiment.

A second small general-text lead is [ManyThings' Tatoeba-derived English–Central Kurdish selection](https://www.manythings.org/anki/), which listed **53** pairs in its 2026-02-13 export. It includes per-sentence attribution, and [Tatoeba's downloads page](https://tatoeba.org/en/downloads) states CC BY 2.0 France for its downloads. Check the actual release, sentence-level authorship, source/translation rights, attribution and quality before use; this is neither a subtitle corpus nor a substantial training set. Do not confuse it with the separately distributed [Tatoeba Challenge](https://github.com/Helsinki-NLP/Tatoeba-Challenge), whose listed training-data license is CC BY-NC-SA 4.0 and whose test/development sets have different conditions.

### 4. CORDI dialogue benchmark — promising but rights unresolved

The CORDI repository describes a corpus of Central Kurdish movie/TV dialogue and states CC BY-SA 4.0 [1](https://github.com/sinaahmadi/CORDI). Its paper describes a separate **300-sentence** English/Standard Kurdish and dialect-parallel set for machine-translation evaluation; the much larger transcription count is **not** 186,038 English–Sorani pairs [2](https://arxiv.org/pdf/2403.01983). The machine-translation documentation is at:

<https://github.com/sinaahmadi/CORDI/blob/main/MT/README.md>

This is a relevant lead for colloquial Sorani, including Erbil and other varieties, but it is not cleared for this commercial project. Its underlying works are films or broadcasts: publicly broadcast does not itself prove permission for commercial training or redistribution. Establish rights in the original dialogue and translations and assess attribution/share-alike obligations. The available independent film grouping, actual subtitle timings, and overlap with candidate models' pretraining have not been verified. Do not use published NLLB-generated outputs as commercial training data by assumption. Even if cleared, a public 300-sentence benchmark would supplement, not replace, a separate authorized, film-disjoint subtitle test.

### 5. Public benchmarks — supplemental evaluation, not subtitle training

- **BOUQuET:** the publisher lists Sorani `ckb_Arab` and CC BY 4.0; the base dataset is intended for evaluation, with `dev` available for validation, not generative-model training [1](https://huggingface.co/datasets/facebook/bouquet). Respect its gated access conditions and attribution, and verify the available Sorani records after authorized access. It is not a timed subtitle test.
- **TICO-19:** the project says its released content is CC0 and includes Sorani [1](https://tico-19.github.io/). The benchmark has existing development and test splits [1](https://tico-19.github.io/testset.html), with 971 and 2,100 sentences respectively [5](https://tico-19.github.io/data/paper/ticopaper.pdf). Its COVID/public-health domain is not representative of film dialogue. Check original English source rights and notices as well as the release terms, preserve its benchmark splits, and use it only as a supplemental evaluation lead in this plan.
- **FLORES+:** its card lists Sorani and CC BY-SA 4.0, says it is for evaluation and **not training**, and requires accepting access conditions [1](https://huggingface.co/datasets/openlanguagedata/flores_plus). Do not represent it as an independent subtitle benchmark.

Any public benchmark may overlap with a candidate model's pretraining. Investigate and report known overlap and remaining uncertainty; public benchmarks cannot establish that the private subtitle test is uncontaminated.

## Commercial-model leads — untested and not cleared

### Option A: two Helsinki-NLP Marian models

The forward model's card lists Apache-2.0, English input and `ckb` output, and requires the `>>ckb<<` target-language prefix [1](https://huggingface.co/Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-iir):

<https://huggingface.co/Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-iir>

The separate reverse model's card lists Apache-2.0, `ckb` among the input languages, and English output:

<https://huggingface.co/Helsinki-NLP/opus-mt-tc-bible-big-iir-en>

Their file listings show a `model.safetensors` file of about **960 MB each** ([forward listing](https://huggingface.co/Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-iir/tree/main), [reverse listing](https://huggingface.co/Helsinki-NLP/opus-mt-tc-bible-big-iir-en/tree/main)), excluding tokenizers, caches and temporary files. This is a disk-file size, not a measured RAM/VRAM requirement or speed estimate. The forward card reports English→Sorani BLEU **7.7** on FLORES-200 devtest; that is the publisher's out-of-domain result, not our test and not a measure of film quality [1](https://huggingface.co/Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-iir). The reverse card's aggregate score does not establish Sorani-specific quality.

The cards identify OPUS/Tatoeba, HPLT and Bible-related training resources, and name `opusTCv20230926max50+bt+jhubc` as the training data. The [Tatoeba Translation Challenge](https://github.com/Helsinki-NLP/Tatoeba-Challenge) describes its **training-data release** as CC BY-NC-SA 4.0, while the model cards label the weights Apache-2.0. This is a material provenance and licensing question for the exact models: establish which data and permissions apply and obtain qualified clarification before commercial use. It does **not**, by itself, prove that the weights inherit the dataset license or that the cards are wrong. The challenge's training release is distinct from its separately described development/test data. Review attribution, redistribution and other upstream obligations as well. Both models still require bilingual quality, privacy, context, resource and app-compatibility testing. Smaller files do not prove faster or better translation.

### Option B: Google MADLAD-400 3B

`google/madlad400-3b-mt` is an Apache-2.0 T5-based multilingual translation checkpoint; its card metadata lists English and `ckb` [3](https://huggingface.co/google/madlad400-3b-mt/blob/main/README.md). Verify both directions and the exact language-tag behavior in the actual tokenizer. Its card says it is a research model not assessed for production, and its underlying web and parallel-data provenance needs review [2](https://huggingface.co/google/madlad400-3b-mt).

The listed full-precision `model.safetensors` file is about **11.8 GB**, excluding tokenizer files, cache copies, workspace and training memory [1](https://huggingface.co/google/madlad400-3b-mt/tree/main). The Windows machine previously reported limited free disk space, but its current storage and RAM/GPU availability are unknown; check them before considering a download. Smaller quantized files exist, but require their own compatible runtime and quality tests and are not assumed to work in this app. Neither the Helsinki candidates nor MADLAD has been downloaded, integrated, or benchmarked by us.

### Optional: external translation service

Google Cloud Translation's **neural machine translation** language list includes English and Kurdish Sorani `ckb` [1](https://docs.cloud.google.com/translate/docs/languages). This is a separate hosted-service option, not a locally trainable checkpoint or an existing app integration. Do not assume `ckb` is supported for every Google custom-training product. Before considering it, review pricing, terms, data retention, customer consent, confidentiality, region, quality, and whether sending subtitle text to a third party is acceptable. Do not send the user's files to an external service without agreement.

## Why a plain CSV is not enough for this app

The current `ml.data.adapters.local` expects timed subtitle files under `root/<film_id>/en.*` and `ckb.*`. `ml/train/train_lora.py` loads a Hugging Face `DatasetDict` with `train`, `val`, and `test`; `ml/train/dataset.py` validates film IDs, start times, provenance and language-verification flags and builds context from earlier source sentences. `ml/eval/records.py` requires reviewed gold IDs from the same held-out dataset and real positive original source durations for subtitle reading-speed metrics. The web **Models & feedback** UI trains on eligible in-app feedback; it does not automatically import an external corpus.

For context, in-app feedback is eligible only when an approved cue has nonempty source and human text and the human text differs from the machine suggestion. For each direction, the training route requires eligible feedback from at least two project IDs for its train/validation split, plus an explicit opt-in because the app has not language-verified those edits. A completed run creates a private, initially inactive `feedback-<job prefix>` adapter; it does not automatically prove quality or commercial rights. Different project IDs are not proof of independent works, and these checks do not provide a separate untouched test set. The current feedback trainer still uses the NLLB-specific implementation, so do not launch it as a commercial solution without resolving the base-model and feedback rights and changing the architecture where required.

Moreover, `ml/inference/nllb_translator.py` and the current trainer encode NLLB-specific language tokens and forced target IDs. Marian and T5 use different tokenizers, language controls, model internals and potentially LoRA target-module names. Changing a model path or renaming CSV columns is not sufficient. A replacement must preserve the app's sentence reconstruction, source context, timing, cue lineage, right-to-left handling, glossary and QA behavior. If a candidate cannot use genuine previous-sentence context adequately, it does not satisfy the project's context-aware requirement merely because it translates isolated sentences.

## Implementation and remaining stages

### Stage 1: rights-aware audit and staging — partially implemented

The [paired-line staging CLI](text-pair-staging.md) supports synthetic demos and explicitly supplied UTF-8 line files. The [TMX staging CLI](tmx-pair-staging.md) separately supports plain English/`ckb` translation units, preserving the unit's reported source-language metadata, original decoded text and ordinal, with whole-unit quarantine for unsupported structures. Both require a source/rights manifest and user attestation, but **do not verify the statements or the source**. They record the original manifest and file hashes, flag exact duplicate/script/length/orthography issues, and write only to a new output directory. Neither downloads data. Matching counts, language tags and flags do not establish Sorani, semantic alignment, or translation quality. Outputs have no invented film IDs or timings and are not trainer-ready. Only independently authored synthetic inputs have been run through these tools by us.

**Still to do, with separate approval as appropriate:**

- Before staging real material, resolve the rights for the relevant original text, translations and intended uses; an attestation or license label alone is not proof. Preserve attribution, restrictions, and evidence, and keep raw data and reports private.
- If approved and needed, add SMOL-style JSONL as a separate input format, validating each `id`, `sl`, `tl`, `src`, `trg` and `is_src_orig` row atomically and retaining original-direction evidence. It is not supported by the current CLI.
- Obtain documented language and bilingual alignment/quality review; assess near-duplicates and cross-source leakage. Do not infer Sorani from Arabic script or a generic `ku` label, or infer contextual order from adjacent text lines.

### Stage 2: an honest text-only data contract — not yet implemented

Extend the schema and trainer so `text_pair` and timed `subtitle_pair` records are distinct. A text-only row needs its real collection and pair identity, provenance, rights/attribution and language-verification evidence; it has no invented film ID, subtitle timestamps, speakers, or CPS. Treat an isolated-pair collection with unknown document grouping as one conservative source group and provide **empty context**. Never make adjacent independent rows into preceding dialogue. Only use document grouping and ordering when supported by genuine metadata.

Update and test validation, grouping, fingerprints, serialization, deduplication, and trainer context construction without weakening the existing timed-subtitle rules. If a text-only evaluator is added, it must omit timed subtitle metrics rather than invent durations. Preserve compatibility with existing timed datasets. A new CLI import path and any web registration workflow would be separate, tested features.

### Stage 3: independent splits and model evaluation

- The new [offline pair-overlap audit](overlap-audit.md) compares a current subtitle `DatasetDict`'s train/validation/test pairs, optional staged text/TMX candidates, and supplied direction-specific corrected gold against held-out pairs. It reports shared film IDs, exact one-sided/bilingual/swapped matches and a bounded deterministic trigram-near signal. It is a separate review tool, tested only on synthetic fixtures, **not an automatic training gate** or proof of independence. It cannot screen quarantined or filtered material, base-model pretraining, different work aliases, paraphrases or reconstructed context. It refuses nonempty explicit `source_context`; omitted gold references are not checked. Matches require human review, and no match does not approve the dataset.
- Before any training, reserve genuinely separate, commercially authorized subtitle works for validation and final test. Use stable source/work IDs, not invented film IDs for text collections. Screen exact and near-duplicate text across splits, including reversed versions of the same pair, and document possible base-model pretraining contamination. Do not call a random line split film-disjoint.
- Freeze and restrict the final test set before tuning. After the relevant rights and evaluation approval, use the independent validation works to measure the eligible base without fine-tuning and to compare any trained candidates. Freeze the selected candidate, evaluation method and acceptance criteria before comparing it with the same base on the untouched test works in **both directions**; do not tune against that final test. Have qualified English–Sorani bilingual reviewers verify references and assess omissions, mistranslations, names, register, dialect, context and consistency, not only automatic scores. Do not assume that training on 863 general sentences will improve results.
- For actual subtitle works, retain real original timings and assess CPS, line lengths, subtitle reconstruction and unchanged timestamps. General-text datasets cannot establish these metrics. Set acceptance criteria before running the final evaluation; report limitations and failures as well as scores.
- The current evaluator ties reviewed gold to the test split of the same `DatasetDict`. Supporting a new mixed/text-only contract and retaining independent authorized timed test works will require code changes and regression tests; no bypass or fabricated values.

### Stage 4: model integration and tests

1. Design a model-specific adapter/tokenization interface for Marian or T5, including correct target-language control, direction, batching, context behavior, output decoding, optional confidence and provenance. Do not manufacture confidence scores when a backend does not provide them.
2. Separately verify training compatibility, LoRA modules, adapter save/reload and base-model identity for the chosen architecture. Do not reuse the NLLB adapter or assume existing NLLB weights can be re-licensed through fine-tuning.
3. Test using independently authored **synthetic fixtures only**: JSONL schema errors and direction flags; mismatched paired lines and blanks; ZWNJ/heh handling; rights gate and safe paths; stable IDs and cross-split leakage; empty text-only context; timed-subtitle and dataset-fingerprint regressions; no fabricated CPS; tokenizer/model-specific language handling; cue timings, source lineage, RTL, glossary and QA behavior.
4. A random offline smoke model proves only software wiring, not translation quality. Download and evaluate a real candidate only after explicit agreement, rights review, and disk/RAM/GPU checks. Do not claim speed or quality improvement until measured.

## Decision gates and operational safeguards

- **Before real data use:** permission for the planned commercial uses, attribution/other obligations, source provenance, independent subtitle evaluation material, and user's informed agreement.
- **Before training:** a cleared compatible base model, validated dataset and separate held-out works, documented bilingual review, hardware/storage budget, and a specific approved run. If any gate fails, stop at planning/audit; do not bypass checks.
- **Before activation or commercial release:** actual results for both directions, review of errors and context handling, subtitle timing/quality tests, security/privacy and deployment review, and explicit approval. No improvement or legal clearance is implied by this plan.
- **Windows:** the existing job may still be running; its current status is unknown. Do not cancel it, restart the worker, rebuild Docker, download weights, or launch training without agreement. A single worker may leave new jobs queued; interruption may lose translation progress.

## Historical inspection of the blocked Interdialect source

For traceability only, the earlier plan inspected the upstream English–Sorani files in memory at commit `a27afa395992c388c777bd0b1dbc0e412484550a`; they were not imported. The two UTF-8 files each had 650 nonblank physical lines. Equal line counts do not establish correct alignment. The inspected XML segments lacked reliable article grouping and subtitle timing; the combined TMX failed standard XML parsing near its beginning. Do not use any of these files in this commercial project without separate permissions.

| File under `CKB-ENG/` | SHA-256 at the inspected commit |
| --- | --- |
| `CKB-ENG.ENG_no_tag.txt` | `25574f3d20fbf83bdc0f717136853e488d67111aa023f44274f5573ca32c81f0` |
| `CKB-ENG.CKB_no_tag.txt` | `6a29f7de1cee8c6d4f9f4657ac19e659d8b361596b6a5f7c1fae3f1be41365d8` |
