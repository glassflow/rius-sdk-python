# Changelog

## [1.0.0](https://github.com/glassflow/rius-sdk-python/compare/v0.17.0...v1.0.0) (2026-09-25)


### ⚠ BREAKING CHANGES

* normalize model_parameters keys onto gen_ai.request.* and rius.request.* ([#117](https://github.com/glassflow/rius-sdk-python/issues/117))
* name spans after their operation and target, per the conventions ([#110](https://github.com/glassflow/rius-sdk-python/issues/110))
* map RETRIEVER to the retrieval operation and take a data source id ([#106](https://github.com/glassflow/rius-sdk-python/issues/106))
* emit rius.span.pending and the rius tracer scope; drop the GLASSFLOW_ env fallback ([#104](https://github.com/glassflow/rius-sdk-python/issues/104))

### Features

* derive both taxonomy keys on spans that carry only one ([#121](https://github.com/glassflow/rius-sdk-python/issues/121)) ([236f6bf](https://github.com/glassflow/rius-sdk-python/commit/236f6bf82ce0aeae1560e760616a593ab76ec95c))
* emit rius.context.sizes, a per-part byte breakdown of the context, on every generation span ([#102](https://github.com/glassflow/rius-sdk-python/issues/102)) ([67556cc](https://github.com/glassflow/rius-sdk-python/commit/67556cc528a3ec9c2eec0aa4bd17286113e30db4))
* emit rius.span.pending and the rius tracer scope; drop the GLASSFLOW_ env fallback ([#104](https://github.com/glassflow/rius-sdk-python/issues/104)) ([c9344c6](https://github.com/glassflow/rius-sdk-python/commit/c9344c61eeb2bc23187a72992e0c5423836c904f))
* fill error.type from the exception event on auto-instrumented spans ([#124](https://github.com/glassflow/rius-sdk-python/issues/124)) ([9546a10](https://github.com/glassflow/rius-sdk-python/commit/9546a10452442f7f1c72a86de7c0d81ffa2bf374))
* gen_ai.tool.name and gen_ai.tool.definitions on OpenInference spans ([#125](https://github.com/glassflow/rius-sdk-python/issues/125)) ([ac1061a](https://github.com/glassflow/rius-sdk-python/commit/ac1061afb51a027ad020f47cbad05dac62840665))
* map llm.finish_reason onto gen_ai.response.finish_reasons ([#122](https://github.com/glassflow/rius-sdk-python/issues/122)) ([7f99d9b](https://github.com/glassflow/rius-sdk-python/commit/7f99d9b1aa8eb83d17f24f6c4c0670b35c99521c))
* map RETRIEVER to the retrieval operation and take a data source id ([#106](https://github.com/glassflow/rius-sdk-python/issues/106)) ([7af1944](https://github.com/glassflow/rius-sdk-python/commit/7af1944f9da07bb91e7ebf05a9207ec539400d52))
* map the cache_creation and details.reasoning_tokens usage spellings onto the canonical counts ([#129](https://github.com/glassflow/rius-sdk-python/issues/129)) ([0932045](https://github.com/glassflow/rius-sdk-python/commit/09320453d426bfc05aa60f0777f3ff5d117f32ff))
* map the OpenInference model-call and usage keys onto the canonical wire ([#118](https://github.com/glassflow/rius-sdk-python/issues/118)) ([3c9bedf](https://github.com/glassflow/rius-sdk-python/commit/3c9bedfc892db1d07d3e3b35cab96ce135cad154))
* name spans after their operation and target, per the conventions ([#110](https://github.com/glassflow/rius-sdk-python/issues/110)) ([7cc18ec](https://github.com/glassflow/rius-sdk-python/commit/7cc18eca61bc6c7d8b44a463c1c424b8466547a3))
* name the agent an AGENT span invokes ([#109](https://github.com/glassflow/rius-sdk-python/issues/109)) ([2362465](https://github.com/glassflow/rius-sdk-python/commit/2362465b5c22f463678eed06144d6b59f095dd53))
* name the agent executing a tool on execute_tool spans ([#112](https://github.com/glassflow/rius-sdk-python/issues/112)) ([6414f66](https://github.com/glassflow/rius-sdk-python/commit/6414f66bcd53cf632e8839478cac945de9fe8efc))
* normalize model_parameters keys onto gen_ai.request.* and rius.request.* ([#117](https://github.com/glassflow/rius-sdk-python/issues/117)) ([02d0608](https://github.com/glassflow/rius-sdk-python/commit/02d0608a10b3e655c88501fcd184828f6796ded8))
* normalize the OpenInference first-token event into gen_ai.first_token ([#116](https://github.com/glassflow/rius-sdk-python/issues/116)) ([ca5b5f1](https://github.com/glassflow/rius-sdk-python/commit/ca5b5f171b4e33a76b60efd3603c469c2a0f1569))
* normalize third-party attribute keys onto the canonical wire ([#113](https://github.com/glassflow/rius-sdk-python/issues/113)) ([5c2f635](https://github.com/glassflow/rius-sdk-python/commit/5c2f6350f418282a5b5be481b575dafbd60aa20c))
* reassemble OpenInference's flattened messages, Anthropic's multi-part contents included ([#133](https://github.com/glassflow/rius-sdk-python/issues/133)) ([2a0cff7](https://github.com/glassflow/rius-sdk-python/commit/2a0cff7dab3db8d2b8a8efda6ba2feb03c11d5d2))
* record the response id, output type, tool call identity and service version ([#114](https://github.com/glassflow/rius-sdk-python/issues/114)) ([978f64a](https://github.com/glassflow/rius-sdk-python/commit/978f64a55dbff6010dad3afa93b63095214c0bce))
* record_exception on the manual span and generation handles ([#107](https://github.com/glassflow/rius-sdk-python/issues/107)) ([186b8e4](https://github.com/glassflow/rius-sdk-python/commit/186b8e44c1a1b43114b6d3bf99d81f822db3fc19))
* take gen_ai.tool.name as an explicit argument, not from the span name ([#105](https://github.com/glassflow/rius-sdk-python/issues/105)) ([f6d2b43](https://github.com/glassflow/rius-sdk-python/commit/f6d2b43f6f7864c419188e30c9a91cb927348855))
* take the main agent identity into rius.main_agent.*, and version the agent at both scopes ([#115](https://github.com/glassflow/rius-sdk-python/issues/115)) ([f358876](https://github.com/glassflow/rius-sdk-python/commit/f358876a083567c31f811d81dd66522c9b9573d8))


### Bug Fixes

* a blank OTEL attribute-count env var no longer counts as the user's limit ([#137](https://github.com/glassflow/rius-sdk-python/issues/137)) ([fff0268](https://github.com/glassflow/rius-sdk-python/commit/fff026861f1e2fd76250affd4878623e6902377b))
* defer to an OTEL attribute-count env var only when OpenTelemetry can use its value ([#139](https://github.com/glassflow/rius-sdk-python/issues/139)) ([890f9a9](https://github.com/glassflow/rius-sdk-python/commit/890f9a93c157cb695ad25289d1c2ee41908cd15f))
* give auto-instrumented embedding spans a request model, and stop exporting their vectors ([#135](https://github.com/glassflow/rius-sdk-python/issues/135)) ([b5a5ba0](https://github.com/glassflow/rius-sdk-python/commit/b5a5ba00c8f6f2a4d922d186a768a7ff43282a73))
* guard native model_parameters with the normalizer's per-key checks, and let the first spelling win ([#131](https://github.com/glassflow/rius-sdk-python/issues/131)) ([3868180](https://github.com/glassflow/rius-sdk-python/commit/3868180d4c793512483b82762b365083dbb47bf8))
* keep tool_choice with content capture off by promoting it out of the request bag ([#132](https://github.com/glassflow/rius-sdk-python/issues/132)) ([f0edb51](https://github.com/glassflow/rius-sdk-python/commit/f0edb5149fba8458c0c21820c5401bb24e37e115))
* raise the span attribute count limit so long agent spans keep their model, tools and system prompt ([#134](https://github.com/glassflow/rius-sdk-python/issues/134)) ([7e6c256](https://github.com/glassflow/rius-sdk-python/commit/7e6c2564dbb50a55acb762c5161c85e5f1e86b98))
* recover gen_ai.response.id from output.value on auto-instrumented LLM spans ([#136](https://github.com/glassflow/rius-sdk-python/issues/136)) ([7f6e7aa](https://github.com/glassflow/rius-sdk-python/commit/7f6e7aa9e5e09ed982537ff274b8d3fd78c4ad90))
* stamp gen_ai.agent.name on the resource ([#108](https://github.com/glassflow/rius-sdk-python/issues/108)) ([7acffb0](https://github.com/glassflow/rius-sdk-python/commit/7acffb05b0da18af7963d6d2b98911625cbd91e9))
* take the response id from output.value only when it is declared JSON or undeclared ([#138](https://github.com/glassflow/rius-sdk-python/issues/138)) ([1622481](https://github.com/glassflow/rius-sdk-python/commit/162248147fbe47e6d574ba6842130fa5c17d0bc5))
* treat the OpenInference request-parameter bag as content ([#120](https://github.com/glassflow/rius-sdk-python/issues/120)) ([8f58973](https://github.com/glassflow/rius-sdk-python/commit/8f5897300ccf072f76b409eb3b156bab0abc9605))
* write JSON-valued span attributes compact and in raw UTF-8 ([#127](https://github.com/glassflow/rius-sdk-python/issues/127)) ([84d5358](https://github.com/glassflow/rius-sdk-python/commit/84d535835b99672675b743bc884cf76af89d1cac))
* write U+2028/U+2029 raw in a promoted tool_choice, as the sink does ([#141](https://github.com/glassflow/rius-sdk-python/issues/141)) ([b9da558](https://github.com/glassflow/rius-sdk-python/commit/b9da558d852d3f1fd3bd638daeb4c176ffc51c9e))

## [0.17.0](https://github.com/glassflow/rius-sdk-python/compare/v0.16.0...v0.17.0) (2026-09-21)


### Features

* emit gen_ai.response.time_to_first_chunk alongside the first-token event ([#100](https://github.com/glassflow/rius-sdk-python/issues/100)) ([a185320](https://github.com/glassflow/rius-sdk-python/commit/a185320f2705b492e255099465ecedc56b12ca21))
* set gen_ai.tool.name on local tool spans ([#97](https://github.com/glassflow/rius-sdk-python/issues/97)) ([31595c3](https://github.com/glassflow/rius-sdk-python/commit/31595c3ac81153f6644b418a2c4d13fdb8ace818))
* set the OTel SpanKind field from the span taxonomy ([#99](https://github.com/glassflow/rius-sdk-python/issues/99)) ([e203d56](https://github.com/glassflow/rius-sdk-python/commit/e203d56041a3d41f72ed2b6b7bf78a590bb8ca3b))


### Bug Fixes

* set error.type on inference and generic spans ([#98](https://github.com/glassflow/rius-sdk-python/issues/98)) ([680f3e0](https://github.com/glassflow/rius-sdk-python/commit/680f3e03751ee6275dc8961911c86e12856d9ccf))

## [0.16.0](https://github.com/glassflow/rius-sdk-python/compare/v0.15.1...v0.16.0) (2026-09-21)


### Features

* bring MCP client spans onto the OTel MCP semantic conventions ([#92](https://github.com/glassflow/rius-sdk-python/issues/92)) ([8e8dd05](https://github.com/glassflow/rius-sdk-python/commit/8e8dd0571079a80e5f5dfd16fd9793bd491d9172))


### Bug Fixes

* set error.type on MCP tool-call spans ([#95](https://github.com/glassflow/rius-sdk-python/issues/95)) ([4c9f8d6](https://github.com/glassflow/rius-sdk-python/commit/4c9f8d61e21d7fba7a242850e24a84ce242ef947))


### Documentation

* mark mcp.result_type as non-semconv and note the mcp 1.x ordering caveat ([#94](https://github.com/glassflow/rius-sdk-python/issues/94)) ([054cc80](https://github.com/glassflow/rius-sdk-python/commit/054cc800bf3679ffae088d9b979ba43be5b77ce8))
* the GenAI execute-tool span says INTERNAL; CLIENT is the MCP-side call ([#96](https://github.com/glassflow/rius-sdk-python/issues/96)) ([659bf71](https://github.com/glassflow/rius-sdk-python/commit/659bf71870fa4db9065087812e88153bdc37174a))

## [0.15.1](https://github.com/glassflow/rius-sdk-python/compare/v0.15.0...v0.15.1) (2026-09-14)


### Bug Fixes

* **masking:** metadata.&lt;key&gt; is content whenever &lt;key&gt; is ([#89](https://github.com/glassflow/rius-sdk-python/issues/89)) ([99a0f41](https://github.com/glassflow/rius-sdk-python/commit/99a0f41f301f140a5324ed68900c21d5082b103d))

## [0.15.0](https://github.com/glassflow/rius-sdk-python/compare/v0.14.0...v0.15.0) (2026-09-14)


### Features

* record tool definitions in the generation API and pin instrumentor tool capture ([#77](https://github.com/glassflow/rius-sdk-python/issues/77)) ([a35373a](https://github.com/glassflow/rius-sdk-python/commit/a35373a67c6b028175bfc8ef26055421ecb1d5e7))
* user() scope stamps user.id on every span in scope ([#80](https://github.com/glassflow/rius-sdk-python/issues/80)) ([01a15c9](https://github.com/glassflow/rius-sdk-python/commit/01a15c9d4432f682d3a286f27483af85258a6530))


### Bug Fixes

* **client:** SDK helpers follow the active client across shutdown() + init() ([#82](https://github.com/glassflow/rius-sdk-python/issues/82)) ([2ca1ea6](https://github.com/glassflow/rius-sdk-python/commit/2ca1ea6f0801d4b6965c6880d0e5bd0e5b0f6133))
* **masking:** sanitize the status description and the GenAI tool-call keys ([#81](https://github.com/glassflow/rius-sdk-python/issues/81)) ([42e5c57](https://github.com/glassflow/rius-sdk-python/commit/42e5c574ad7c3970dd0bed109c242aaacb8ad4d1))
* robustness edge cases from the 2026-09-14 review ([#87](https://github.com/glassflow/rius-sdk-python/issues/87)) ([90acd2c](https://github.com/glassflow/rius-sdk-python/commit/90acd2ce2be43b8c4fc9db2c4a1e42a571430657))
* **serde:** 32 KB bound that stops encoding at the cap; identity at span start for observe and MCP ([#86](https://github.com/glassflow/rius-sdk-python/issues/86)) ([7f9c5dc](https://github.com/glassflow/rius-sdk-python/commit/7f9c5dccd59f5d89f1eb3b3fb5f4891f96cb9f9c))
* strip tool definitions and Vercel ai.* content on every sanitized path ([#79](https://github.com/glassflow/rius-sdk-python/issues/79)) ([000bd53](https://github.com/glassflow/rius-sdk-python/commit/000bd53932c372e3587b3b83d886f0870e31c985))

## [0.14.0](https://github.com/glassflow/rius-sdk-python/compare/v0.13.0...v0.14.0) (2026-09-10)


### Features

* capture reasoning output tokens in set_usage ([#73](https://github.com/glassflow/rius-sdk-python/issues/73)) ([7244add](https://github.com/glassflow/rius-sdk-python/commit/7244add11ec4e49d321f5a2a90db44e5c90f33f1))
* capture requested reasoning effort level ([#75](https://github.com/glassflow/rius-sdk-python/issues/75)) ([141d7c3](https://github.com/glassflow/rius-sdk-python/commit/141d7c3e580eff48af1bec5f4590f3cad0d7966f))


### Bug Fixes

* sum Anthropic cache tokens into the emitted input-token total ([#76](https://github.com/glassflow/rius-sdk-python/issues/76)) ([36795f9](https://github.com/glassflow/rius-sdk-python/commit/36795f97c05028612317c5262c97931adc9e16c2))

## [0.13.0](https://github.com/glassflow/rius-sdk-python/compare/v0.12.0...v0.13.0) (2026-09-10)


### Features

* add cache-token fields to set_usage ([#65](https://github.com/glassflow/rius-sdk-python/issues/65)) ([e1682d1](https://github.com/glassflow/rius-sdk-python/commit/e1682d1f93906134eec7ccdeddb1cd5f44c9f157))
* context-scoped multi-workspace routing (workspace() scope + routing exporter) ([#68](https://github.com/glassflow/rius-sdk-python/issues/68)) ([621fab2](https://github.com/glassflow/rius-sdk-python/commit/621fab2c3e98ce7e8619850bce9dc5e877bac458))


### Bug Fixes

* emit gen_ai.usage.cache_write.input_tokens per the semconv rename ([#69](https://github.com/glassflow/rius-sdk-python/issues/69)) ([2b91c90](https://github.com/glassflow/rius-sdk-python/commit/2b91c90094e545f830dfccbca6a33c7b6a023f31))
* strip tool definitions and descriptions when content capture is off ([#71](https://github.com/glassflow/rius-sdk-python/issues/71)) ([6453658](https://github.com/glassflow/rius-sdk-python/commit/64536585edf6ecf97de5b570784b283ac3b96520))

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
