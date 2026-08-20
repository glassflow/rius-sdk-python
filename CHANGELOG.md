# Changelog

## [0.12.0](https://github.com/glassflow/rius-sdk-python/compare/v0.11.0...v0.12.0) (2026-08-20)


### Features

* session ids — a scoped session() plus an init-level default ([#61](https://github.com/glassflow/rius-sdk-python/issues/61)) ([4279e43](https://github.com/glassflow/rius-sdk-python/commit/4279e43687dbf47c4d5746c07f857d6d26b5929c))
* stamp service.instance.id on spans and share it with the heartbeat ([#63](https://github.com/glassflow/rius-sdk-python/issues/63)) ([e0b2ae9](https://github.com/glassflow/rius-sdk-python/commit/e0b2ae90be997ef9045486e8b17b191488dd61bc))

## [0.11.0](https://github.com/glassflow/rius-sdk-python/compare/v0.10.0...v0.11.0) (2026-08-17)


### ⚠ BREAKING CHANGES

* drop the mcp extra so every extra means one thing ([#59](https://github.com/glassflow/rius-sdk-python/issues/59))

### Features

* drop the mcp extra so every extra means one thing ([#59](https://github.com/glassflow/rius-sdk-python/issues/59)) ([0672e94](https://github.com/glassflow/rius-sdk-python/commit/0672e94d4cbae0a21dc886f4bf7dd097db108975))

## [0.10.0](https://github.com/glassflow/rius-sdk-python/compare/v0.9.1...v0.10.0) (2026-08-13)


### ⚠ BREAKING CHANGES

* enable the heartbeat by default ([#57](https://github.com/glassflow/rius-sdk-python/issues/57))

### Features

* enable the heartbeat by default ([#57](https://github.com/glassflow/rius-sdk-python/issues/57)) ([fe08e46](https://github.com/glassflow/rius-sdk-python/commit/fe08e4612dd7e665088f90b4296951574b624ca2))

## [0.9.1](https://github.com/glassflow/rius-sdk-python/compare/v0.9.0...v0.9.1) (2026-08-12)


### Bug Fixes

* cover bare llm.prompts and llm.prompt_template in content controls ([#54](https://github.com/glassflow/rius-sdk-python/issues/54)) ([82dedf7](https://github.com/glassflow/rius-sdk-python/commit/82dedf7406b7f80466178a730e922017a792820a))
* sanitize span events and links, not just span attributes ([#56](https://github.com/glassflow/rius-sdk-python/issues/56)) ([2caa7f1](https://github.com/glassflow/rius-sdk-python/commit/2caa7f1f4a1ee78053978de8e6aca04d2f068406))

## [0.9.0](https://github.com/glassflow/rius-sdk-python/compare/v0.8.2...v0.9.0) (2026-08-12)


### Features

* RIUS_-prefixed env vars with deprecated GLASSFLOW_ aliases ([#53](https://github.com/glassflow/rius-sdk-python/issues/53)) ([893e265](https://github.com/glassflow/rius-sdk-python/commit/893e2655efce9d19dbb7e941d08f51573e7d23ab))
* surface export failures — init warnings, connectivity check, honest flush ([#51](https://github.com/glassflow/rius-sdk-python/issues/51)) ([7138f45](https://github.com/glassflow/rius-sdk-python/commit/7138f4504aeaaad8a11e5f2577c57a5fe97b5902))

## [0.8.2](https://github.com/glassflow/rius-sdk-python/compare/v0.8.1...v0.8.2) (2026-08-07)


### Bug Fixes

* package author email points at the real support address ([#49](https://github.com/glassflow/rius-sdk-python/issues/49)) ([4da0e10](https://github.com/glassflow/rius-sdk-python/commit/4da0e10d13964e7c0e646c58b27065f2605264fe))

## [0.8.1](https://github.com/glassflow/rius-sdk-python/compare/v0.8.0...v0.8.1) (2026-08-06)


### Bug Fixes

* README documented a dead default endpoint ([#47](https://github.com/glassflow/rius-sdk-python/issues/47)) ([abaadc5](https://github.com/glassflow/rius-sdk-python/commit/abaadc595bace230d7ce37919edcb248f11ed577))

## [0.8.0](https://github.com/glassflow/rius-sdk-python/compare/v0.7.0...v0.8.0) (2026-08-05)


### ⚠ BREAKING CHANGES

* rebrand SDK to glassflow-rius, import rius ([#44](https://github.com/glassflow/rius-sdk-python/issues/44))

### Features

* rebrand SDK to glassflow-rius, import rius ([#44](https://github.com/glassflow/rius-sdk-python/issues/44)) ([82c91c1](https://github.com/glassflow/rius-sdk-python/commit/82c91c1f644ca31e53bc2737e8cf2326dfa1aba6))


### Documentation

* update readme title ([#46](https://github.com/glassflow/rius-sdk-python/issues/46)) ([4ae480a](https://github.com/glassflow/rius-sdk-python/commit/4ae480a626b4175789bac652bf9767bcbe50a705))

## [0.7.0](https://github.com/glassflow/glassflow-python/compare/v0.6.0...v0.7.0) (2026-08-04)


### Features

* emit partial (pending) spans at span start ([#37](https://github.com/glassflow/glassflow-python/issues/37)) ([e39e555](https://github.com/glassflow/glassflow-python/commit/e39e55545bb718ab83c196c5ec5b2ce631c00cb8))


### Bug Fixes

* support mcp 2.x result shapes in MCP instrumentation ([#42](https://github.com/glassflow/glassflow-python/issues/42)) ([65aff4b](https://github.com/glassflow/glassflow-python/commit/65aff4b51bcbd9688f72dd0f0733dbc34bfb642d))


### Documentation

* remove em dashes from docstrings ([#39](https://github.com/glassflow/glassflow-python/issues/39)) ([7ca0af5](https://github.com/glassflow/glassflow-python/commit/7ca0af5a6237fec0918ab8745067da2e1f798004))

## [0.6.0](https://github.com/glassflow/glassflow-python/compare/v0.5.0...v0.6.0) (2026-07-20)


### Features

* agent-lifetime heartbeat sender ([#32](https://github.com/glassflow/glassflow-python/issues/32)) ([7c592a8](https://github.com/glassflow/glassflow-python/commit/7c592a8822341e06b9bcd444502530a84dc80679))

## [0.5.0](https://github.com/glassflow/glassflow-python/compare/v0.4.1...v0.5.0) (2026-07-15)


### Features

* record time-to-first-token on streaming generations ([#29](https://github.com/glassflow/glassflow-python/issues/29)) ([6c1ecaa](https://github.com/glassflow/glassflow-python/commit/6c1ecaa46c7086b2bbdccc5a4d4dd7ade036c3c7))

## [0.4.1](https://github.com/glassflow/glassflow-python/compare/v0.4.0...v0.4.1) (2026-07-09)


### Documentation

* complete Google-style docstrings for the public API ([#26](https://github.com/glassflow/glassflow-python/issues/26)) ([75d82a1](https://github.com/glassflow/glassflow-python/commit/75d82a19617f89cebdc813f61757fdc3a5119b46))

## [0.4.0](https://github.com/glassflow/glassflow-python/compare/v0.3.0...v0.4.0) (2026-07-06)


### Features

* first-class MCP tool-call instrumentation ([#24](https://github.com/glassflow/glassflow-python/issues/24)) ([623004f](https://github.com/glassflow/glassflow-python/commit/623004fdd388b7244d7bbc4374b1023ea7680793))

## [0.3.0](https://github.com/glassflow/glassflow-python/compare/v0.2.0...v0.3.0) (2026-07-05)


### ⚠ BREAKING CHANGES

* Generation.set_model() is now set_response_model(); Generation.set_finish_reason() is now set_finish_reasons().

### Features

* bundled auto-instrumentation via OpenInference ([6945ab1](https://github.com/glassflow/glassflow-python/commit/6945ab13bead44f61d765273e23d9ce26513d6e2))
* pre-1.0 API cleanups from the SDK review ([a294b94](https://github.com/glassflow/glassflow-python/commit/a294b945a01c68da3f918a5e9721b5d67a27433c))


### Bug Fixes

* crash-proofing and semconv corrections ([dd06b18](https://github.com/glassflow/glassflow-python/commit/dd06b187b2396e2648dc60b690f4c6c1367b72bb))
* define init() lifecycle semantics ([4374298](https://github.com/glassflow/glassflow-python/commit/4374298615b5b08920a2b3bda96f4bdd27d760c9))
* emit gen_ai.*.messages in the spec role/parts shape ([a7225d0](https://github.com/glassflow/glassflow-python/commit/a7225d0987c5e63e645b7f5d0f074bb1a42ffd46))
* harden export-stage masking ([9781380](https://github.com/glassflow/glassflow-python/commit/9781380d68936a43b91db45403511d5b2fceeeb7))

## [0.2.0](https://github.com/glassflow/glassflow-python/compare/v0.1.0...v0.2.0) (2026-07-03)


### ⚠ BREAKING CHANGES

* start_generation/start_as_current_generation param 'system' is now 'provider', and the emitted attribute is gen_ai.provider.name (was gen_ai.system).

### Features

* emit gen_ai.provider.name; rename generation param system -&gt; provider ([f30ea71](https://github.com/glassflow/glassflow-python/commit/f30ea71674fe4d2797e9ab5d5842e393b40054c0))
* harden export pipeline reliability ([0934973](https://github.com/glassflow/glassflow-python/commit/093497379eaed0ae0b0d56c9653efc1f37438ed1))
* head-based sampling via sample_rate ([67d4fd1](https://github.com/glassflow/glassflow-python/commit/67d4fd1ea00950645866e391f3c38c6dcbf9cb8b))
* PII masking and content opt-out at export ([2c3b4b0](https://github.com/glassflow/glassflow-python/commit/2c3b4b054bb78e2bc1757e55b944ee8e636aa418))

## [0.1.0](https://github.com/glassflow/glassflow-python/compare/v0.0.1...v0.1.0) (2026-07-02)


### Features

* add `@observe` decorator for tracing user functions ([4e0ba4d](https://github.com/glassflow/glassflow-python/commit/4e0ba4d1013bb527df655012010fcd7accf65004))
* add span-kind model (semconv) and kind param to `@observe` ([e1305d8](https://github.com/glassflow/glassflow-python/commit/e1305d8c05ca03b56975a67645a9c6f004c5a339))
* add start_generation LLM capture helper (gen_ai-native) ([415b168](https://github.com/glassflow/glassflow-python/commit/415b1685403ff2fa6bc4450fe0242e6cd0f5c9a8))
* add start_span manual span API + Observation handle ([70835f8](https://github.com/glassflow/glassflow-python/commit/70835f890ba2e48de22af99b393a9269d27cec1b))
* align span API naming + add manual create/update/end lifecycle ([17e8f31](https://github.com/glassflow/glassflow-python/commit/17e8f31e634963f97c53e2be1fed84d434e85b15))


### Documentation

* update README title to GlassFlow Python SDK ([33cdacb](https://github.com/glassflow/glassflow-python/commit/33cdacb67b11454c16febfccbc89bdc0593bcd18))
