# Smoothed top-two DMRL gain on token-counted alignment data

This exploratory plug-in restores the pasted theoretical gain formula: at each n>=4, use the top two utilities above the third, average those excesses over prefixes 4..n, and multiply by 4/n. It retains the practical beta=2 estimated next-token cost and uses sigmoid(reward - observed-prefix q99) as the utility transform. The reference moves as the prefix grows, token costs vary, and the replay has a cap of 960; the theorem therefore does not directly apply.

Both policies use the same recorded responses, eight response orders, five 50/50 prompt splits, and one fixed N selected on training prompts for each generator/reward/price. Reported gains average splitwise relative net-utility improvements.

| Reward model | Generator | $/M tokens | Fixed N | Theory-gain samples | Current samples | Theory-gain net utility | Current net utility | Theory-gain vs fixed | Current vs fixed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fsfairx_llama3_rm | Gemma 3 4B | 0.1 | 515.8 | 928.47 | 510.96 | 0.5505 | 0.5613 | -1.40% | +0.55% |
| fsfairx_llama3_rm | Gemma 3 4B | 0.2 | 271.4 | 807.50 | 335.41 | 0.4941 | 0.5348 | -6.57% | +1.11% |
| fsfairx_llama3_rm | Gemma 3 4B | 0.5 | 133.4 | 549.16 | 171.95 | 0.4163 | 0.4943 | -13.98% | +2.13% |
| fsfairx_llama3_rm | Gemma 3 4B | 1 | 67.2 | 366.98 | 103.50 | 0.3615 | 0.4611 | -18.77% | +3.62% |
| fsfairx_llama3_rm | Gemma 3 4B | 2 | 38.8 | 233.58 | 57.55 | 0.3080 | 0.4199 | -24.29% | +3.22% |
| fsfairx_llama3_rm | Gemma 3 4B | 5 | 15.6 | 121.70 | 25.64 | 0.2286 | 0.3571 | -34.48% | +2.40% |
| fsfairx_llama3_rm | Gemma 3 4B | 10 | 9.0 | 69.01 | 14.00 | 0.1590 | 0.3121 | -46.71% | +4.67% |
| fsfairx_llama3_rm | Granite 4.2 8B | 0.1 | 411.2 | 923.53 | 428.35 | 0.5335 | 0.5681 | -4.38% | +1.83% |
| fsfairx_llama3_rm | Granite 4.2 8B | 0.2 | 219.8 | 772.25 | 291.01 | 0.4686 | 0.5350 | -10.51% | +2.16% |
| fsfairx_llama3_rm | Granite 4.2 8B | 0.5 | 99.4 | 520.59 | 159.83 | 0.3828 | 0.4862 | -18.90% | +3.01% |
| fsfairx_llama3_rm | Granite 4.2 8B | 1 | 58.6 | 362.29 | 95.51 | 0.3103 | 0.4427 | -27.76% | +3.08% |
| fsfairx_llama3_rm | Granite 4.2 8B | 2 | 32.4 | 233.80 | 52.74 | 0.2361 | 0.3925 | -37.16% | +4.46% |
| fsfairx_llama3_rm | Granite 4.2 8B | 5 | 12.2 | 120.79 | 23.95 | 0.1362 | 0.3226 | -56.95% | +2.01% |
| fsfairx_llama3_rm | Granite 4.2 8B | 10 | 8.0 | 68.09 | 13.34 | 0.0449 | 0.2719 | -82.75% | +4.50% |
| fsfairx_llama3_rm | Ministral 3 8B | 0.1 | 425.8 | 929.11 | 489.31 | 0.5483 | 0.5751 | -2.83% | +1.93% |
| fsfairx_llama3_rm | Ministral 3 8B | 0.2 | 251.2 | 805.76 | 315.54 | 0.4861 | 0.5429 | -8.60% | +2.10% |
| fsfairx_llama3_rm | Ministral 3 8B | 0.5 | 131.8 | 564.66 | 159.20 | 0.3995 | 0.4960 | -16.55% | +3.62% |
| fsfairx_llama3_rm | Ministral 3 8B | 1 | 61.0 | 381.10 | 95.73 | 0.3359 | 0.4535 | -22.62% | +4.45% |
| fsfairx_llama3_rm | Ministral 3 8B | 2 | 33.2 | 241.32 | 53.53 | 0.2732 | 0.4130 | -29.79% | +6.13% |
| fsfairx_llama3_rm | Ministral 3 8B | 5 | 14.0 | 121.22 | 24.92 | 0.1824 | 0.3428 | -43.93% | +5.37% |
| fsfairx_llama3_rm | Ministral 3 8B | 10 | 8.8 | 73.06 | 14.00 | 0.1014 | 0.2910 | -62.77% | +6.97% |
| fsfairx_llama3_rm | Qwen 3.5 9B | 0.1 | 482.0 | 929.59 | 550.10 | 0.5424 | 0.5627 | -1.71% | +1.96% |
| fsfairx_llama3_rm | Qwen 3.5 9B | 0.2 | 231.4 | 802.83 | 368.15 | 0.4908 | 0.5372 | -6.01% | +2.87% |
| fsfairx_llama3_rm | Qwen 3.5 9B | 0.5 | 103.4 | 562.05 | 193.01 | 0.4202 | 0.4994 | -13.07% | +3.31% |
| fsfairx_llama3_rm | Qwen 3.5 9B | 1 | 66.4 | 396.13 | 108.51 | 0.3597 | 0.4649 | -19.85% | +3.61% |
| fsfairx_llama3_rm | Qwen 3.5 9B | 2 | 32.6 | 254.19 | 59.74 | 0.3021 | 0.4243 | -26.69% | +2.97% |
| fsfairx_llama3_rm | Qwen 3.5 9B | 5 | 18.6 | 125.45 | 26.97 | 0.2174 | 0.3612 | -38.11% | +2.86% |
| fsfairx_llama3_rm | Qwen 3.5 9B | 10 | 10.2 | 72.36 | 14.87 | 0.1456 | 0.3191 | -50.90% | +7.77% |
| rm_mistral_7b | Gemma 3 4B | 0.1 | 501.0 | 933.75 | 512.64 | 0.5369 | 0.5535 | -2.35% | +0.66% |
| rm_mistral_7b | Gemma 3 4B | 0.2 | 240.0 | 805.64 | 334.85 | 0.4810 | 0.5306 | -8.02% | +1.46% |
| rm_mistral_7b | Gemma 3 4B | 0.5 | 117.2 | 525.20 | 174.31 | 0.4117 | 0.4905 | -14.46% | +1.93% |
| rm_mistral_7b | Gemma 3 4B | 1 | 65.0 | 339.79 | 103.62 | 0.3638 | 0.4557 | -18.53% | +2.05% |
| rm_mistral_7b | Gemma 3 4B | 2 | 35.0 | 217.20 | 56.44 | 0.3110 | 0.4182 | -23.68% | +2.63% |
| rm_mistral_7b | Gemma 3 4B | 5 | 16.2 | 110.84 | 24.85 | 0.2360 | 0.3591 | -33.51% | +1.17% |
| rm_mistral_7b | Gemma 3 4B | 10 | 8.0 | 61.00 | 13.48 | 0.1753 | 0.3171 | -43.13% | +2.93% |
| rm_mistral_7b | Granite 4.2 8B | 0.1 | 373.2 | 934.61 | 423.26 | 0.5436 | 0.5660 | -3.95% | +0.02% |
| rm_mistral_7b | Granite 4.2 8B | 0.2 | 214.2 | 814.63 | 293.00 | 0.4645 | 0.5317 | -12.05% | +0.68% |
| rm_mistral_7b | Granite 4.2 8B | 0.5 | 124.8 | 540.86 | 159.70 | 0.3680 | 0.4802 | -21.55% | +2.37% |
| rm_mistral_7b | Granite 4.2 8B | 1 | 58.8 | 369.78 | 94.69 | 0.2933 | 0.4359 | -31.22% | +2.21% |
| rm_mistral_7b | Granite 4.2 8B | 2 | 33.4 | 230.57 | 52.09 | 0.2185 | 0.3873 | -42.05% | +2.73% |
| rm_mistral_7b | Granite 4.2 8B | 5 | 13.0 | 113.13 | 23.67 | 0.1178 | 0.3106 | -61.85% | +0.66% |
| rm_mistral_7b | Granite 4.2 8B | 10 | 7.4 | 63.74 | 13.01 | 0.0330 | 0.2552 | -86.58% | +4.14% |
| rm_mistral_7b | Ministral 3 8B | 0.1 | 538.6 | 946.52 | 482.68 | 0.5596 | 0.5800 | -2.32% | +1.24% |
| rm_mistral_7b | Ministral 3 8B | 0.2 | 256.2 | 836.15 | 309.97 | 0.4861 | 0.5451 | -9.04% | +2.02% |
| rm_mistral_7b | Ministral 3 8B | 0.5 | 120.2 | 582.63 | 162.14 | 0.3900 | 0.4921 | -19.23% | +1.91% |
| rm_mistral_7b | Ministral 3 8B | 1 | 62.8 | 391.08 | 98.02 | 0.3200 | 0.4514 | -26.48% | +3.68% |
| rm_mistral_7b | Ministral 3 8B | 2 | 35.6 | 241.58 | 55.65 | 0.2557 | 0.4005 | -34.17% | +3.13% |
| rm_mistral_7b | Ministral 3 8B | 5 | 17.2 | 119.08 | 25.48 | 0.1634 | 0.3289 | -48.93% | +2.82% |
| rm_mistral_7b | Ministral 3 8B | 10 | 8.8 | 69.44 | 14.28 | 0.0823 | 0.2785 | -68.44% | +7.13% |
| rm_mistral_7b | Qwen 3.5 9B | 0.1 | 531.6 | 942.13 | 536.37 | 0.5493 | 0.5683 | -2.00% | +1.39% |
| rm_mistral_7b | Qwen 3.5 9B | 0.2 | 279.0 | 838.28 | 360.27 | 0.4914 | 0.5405 | -7.54% | +1.70% |
| rm_mistral_7b | Qwen 3.5 9B | 0.5 | 112.2 | 591.12 | 189.74 | 0.4104 | 0.5015 | -15.96% | +2.71% |
| rm_mistral_7b | Qwen 3.5 9B | 1 | 73.4 | 410.24 | 106.60 | 0.3501 | 0.4628 | -22.33% | +2.70% |
| rm_mistral_7b | Qwen 3.5 9B | 2 | 35.6 | 255.84 | 58.72 | 0.2907 | 0.4197 | -29.37% | +2.00% |
| rm_mistral_7b | Qwen 3.5 9B | 5 | 17.2 | 121.34 | 26.39 | 0.2103 | 0.3585 | -39.77% | +2.71% |
| rm_mistral_7b | Qwen 3.5 9B | 10 | 9.0 | 69.86 | 14.59 | 0.1408 | 0.3108 | -52.27% | +5.46% |

The earlier near-direct historical ablation did not include the pasted formula's time average, so it is not this experiment.
