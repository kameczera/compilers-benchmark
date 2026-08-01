# Reviews

## Review 1

**Reviewer expertise:** 3: I am an expert on this topic (know the related work well); 2: I am knowledgeable on this topic; 1: I am an informed outsider.

### Paper summary

As I could infer, the aut





/hors propose to formalize the trade-off between compilation-time and execution-time gains during Operator Fusion passes in modern machine learning compilers. Specifically they presented a unified protocol for evaluating machine learning compilers such as TorchInductor, XLA, and TVM, and concluded that:

- TVM exhibits higher granularity (more kernels) and more conservative fusions;
- TorchInductor combines Triton kernels with external library calls; and
- XLA performs more aggressive fusions on Transformers but retains custom calls.

### Strengths

The paper addresses a important and up-to-date research topic.

### Weaknesses

- The paper presentation is too dense, difficult to be read even by a specialist in the compilation fusion.
- The notation and the presented methodology should have been clearly presented to help the paper readers to follow the arguments.

### Detailed comments

- The addressed topic is not new, but it is still relevant for the programming language area.
- The presentation is well structured, but the text is too dense, so I had difficulties to understand the necessary details to reproduce the results.

**Overall merit:** 1: Reject; 2: Weak Reject; 3: Weak Accept; 4: Accept.

---

## Review 2

**Reviewer expertise:** 3: I am an expert on this topic (know the related work well); 2: I am knowledgeable on this topic; 1: I am an informed outsider.

### Paper summary

The paper presents a benchmark of three machine learning compilers to evaluate their operator fusion efficiency. Its main objective is to analyze the trade-off between compilation overhead and execution performance across the three backends. The evaluation is conducted on four ML models, and the authors propose an algebraic framework for backend comparison. The main contributions of the work lie in the benchmark, the experimental pipeline, and the comparison framework.

### Strengths

- The experimental methodology is clear and detailed.
- The results are comprehensive and robust.
- The analysis of the results is thorough and well supported.
- The reference code is provided.

### Weaknesses

- The contributions are relatively minor. The paper mainly presents a benchmark along with a simple standardized metric for backend comparison.
- The text in several figures is too small.
- The text in Figure 5 is in Portuguese.
- The related work section is rather brief and could be better integrated into the introduction.
- Not all figures are referenced in the main text. Figures that are not discussed should either be removed or explicitly incorporated into the analysis.
- Some figures would benefit from more detailed captions.

### Detailed comments

In general, the work is well executed. The methodology is clear, the results are thoroughly explored, and the paper is technically rigorous. Only minor presentation-related corrections are suggested. My main concern lies in the novelty of the contribution, which is relatively limited, as pointed out in the weaknesses section. However, considering the solid methodology and the quality of the benchmark analysis, I believe the work is well suited for SBLP.

**Overall merit:** 1: Reject; 2: Weak Reject; 3: Weak Accept; 4: Accept.

---

## Review 3

**Reviewer expertise:** 3: I am an expert on this topic (know the related work well); 2: I am knowledgeable on this topic; 1: I am an informed outsider.

### Paper summary

This paper for the UG Track experimentally benchmarks three modern compilation toolchains for machine learning kernels. The focus is in analyzing kernel fusion optimization in four reference neural network architectures and measuring the trade-off in compile-time computation and execution efficiency. The paper proposes a explicit linear model to formalize this trade-off, aiming to find the cutting points where each compiler is preferable.

### Strengths

- The paper is well written and very clear in its methodology, which seems to make possible its reproduction.
- Moreover, reference code is provided for the benchmarks.
- The overall comparison is well-grounded and limitations are adequately presented together with the conclusions.
- The experimental comparison is detailed and well discussed.

### Weaknesses

I have only two minor points to note:

- The Related Work section is short but seems to be adequate for the paper, but it would be better placed right after the Introduction.
- Figure 5 has Portuguese text in the displayed graph.

### Detailed comments

I believe this paper is spot-on, covering a very important topic in recent compiler design. It addressed very well the issues concerning kernel fusion and optimization for machine learning models. Both the contributions and the technical depth are, for me, more than enough for a UG Track work.

**Overall merit:** 1: Reject; 2: Weak Reject; 3: Weak Accept; 4: Accept.

---

## Review 4

**Reviewer expertise:** 3: I am an expert on this topic (know the related work well); 2: I am knowledgeable on this topic; 1: I am an informed outsider.

### Paper summary

The paper proposes a framework to evaluate neural network optimization regarding compile and execution times using operator fusion. It also considers fold transformation before the fusion step for compilation and execution improvements. Furthermore, the evaluation is conducted using three compilers on real-world models to evaluate the results.

### Strengths

- A major strength of this paper is that the proposed framework and all experimental evaluations are entirely based on real-world models.
- The paper presents a repository for the source code.
- The the statistical data of execution time are significant.

### Weaknesses

- A significant weakness of the paper is the lack of literature support; specifically, the theoretical background section entirely lacks citations, and other parts of the text are under-referenced.
- The measurement of the compilation time is unclear (It should be informed on text).
- You must be careful with some strong statements such as "Our work argues that the fold operation should be performed before operator fusion, since it simplifies the graph before codegen.".

### Detailed comments

The work contributes to the field of machine learning model compilers. However, there are several adjustments that must be made:

#### Text and Formatting Adjustments

- **Theoretical Background:** This section completely lacks references. You MUST provide appropriate citations.
- **Figure 2:** This figure is not cited anywhere in the text. If it is unnecessary, it should be removed; otherwise, you should explain the rationale for evaluating external kernels.
- **Terminology:** Several terms are not properly defined. What are "internal" and "external" kernels in this context?
- **Notation:** The tuple fields in the residual network inputs (N, C, H, W) are not explained. While H and W likely mean height and width, what are the meanings of N and C?
- **Abbreviations:** Some abbreviations (e.g., GEMM and QKV) are not defined upon first mention.
- **Tables 5 and 6:** These tables are missing text citations/references.
- **Figure 5:** The captions/subtitles in Figure 5 are currently in Portuguese and must be translated to English.
- **Level of Abstraction:** The text contains an excessive number of function names, file names, and internal variables. Please avoid this and focus on a higher-level explanation.

#### Methodology and Discussion Concerns

- **Fold Operation Argument (Page 6):** The authors state: "Our work argues that the fold operation should be performed before operator fusion, since it simplifies the graph before codegen." Have you measured the neural network's accuracy (or other relevant machine learning metrics) to support this conclusion?
- **Experimental Setup (Compilation and Execution Time):** The work focuses on compile and execution times. While the authors conducted multiple executions (50 times), the number of compilation runs is not mentioned. When measuring compile time, you must perform multiple compilations to report the same statistics as execution time. This would help address anomalies such as the one in Table 1 for the TVM compiler. For all inputs except (1,3,512,512), there is no significant time variation between ResNet-18 and ResNet-50. However, this specific input takes almost twice as long to compile ResNet-50. This anomaly should be explained in the text. I strongly suggest conducting multiple compilations to accurately measure compile time and mitigate such issues.

**Overall merit:** 1: Reject; 2: Weak Reject; 3: Weak Accept; 4: Accept.
