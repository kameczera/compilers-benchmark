# Mudanças da versão antiga para a versão nova do artigo

## 1. Escopo deste documento

Este documento compara:

- a versão antiga do artigo: [`old.tex`](old.tex);
- a versão nova: [`main.tex`](main.tex);
- as observações dos revisores: [`reviews.md`](reviews.md).

O objetivo não é reproduzir todas as diferenças de palavras e formatação. O
documento registra todas as mudanças que alteram a metodologia, os resultados,
as conclusões, a interpretação técnica ou a reprodutibilidade do trabalho.

## 2. Resumo em linguagem simples

A versão antiga apresentava resultados interessantes, mas tratava algumas
medições como se fossem mais fortes do que o protocolo permitia. Em especial:

- a compilação era observada essencialmente uma vez por configuração;
- as 50 execuções posteriores mediam apenas a variação dentro de um processo;
- algumas comparações entre backends eram tratadas como rankings diretos,
  embora os backends recebessem grafos e contratos diferentes;
- a redução de tempo atribuída ao fold do BERT não vinha de uma transformação
  real do modelo;
- o texto afirmava que o TorchInductor usava uma ordem de passes ineficiente,
  embora o experimento não isolasse a ordem interna dos passes.

A nova versão corrige esses problemas. As ResNets foram medidas novamente em
cinco processos frios independentes por configuração. Os autores passaram a
calcular incerteza no nível correto, aplicaram testes de Welch e correção de
Holm ao estudo do fold, mediram o custo do próprio preprocessamento e auditaram
a correção numérica da transformação.

O resultado principal ficou mais específico:

> O fold offline de `Conv–BatchNorm` reduz o custo de compilação do
> TorchInductor nas ResNets avaliadas, mas isso não prova que existe uma ordem
> universalmente correta para os passes de compilação.

## 3. Visão geral das principais mudanças

| Tema | Versão antiga | Versão nova |
|---|---|---|
| Unidade experimental | Uma compilação e 50 execuções no mesmo processo | Cinco compilações em processos frios independentes |
| Interpretação das 50 execuções | Podiam aparentar 50 repetições experimentais | São observações internas; cada processo fornece uma média |
| Estatística do fold | Comparação de percentuais pontuais | Welch bilateral sobre cinco médias e correção de Holm |
| Custo do fold | Não era incorporado claramente à decisão | É medido separadamente e somado ao custo fixo do fold |
| Correção do fold ResNet | Argumento algébrico | Argumento + `allclose` + remoção de todas as BatchNorms |
| Fold do BERT | Relatado como grande ganho | Retirado: o matcher fez zero transformações |
| BERT e GPT-2 cross-backend | Tratados como comparações de desempenho | Mantidos apenas como dados históricos descritivos |
| Contagem de kernels | Próxima de um ranking de fusion | Censo estrutural estático, sem equivalência entre IRs |
| Justiça entre backends | “Configurações equivalentes” | Mesmas condições externas, mas pilhas nativas diferentes |
| Tempo de compilação | Chamado diretamente de tempo de compilação | Chamado de proxy específico de cada backend |
| Conclusão sobre passes | TorchInductor teria ordem subótima | Não se estabelece uma ordem universal de passes |
| Limitações | Curtas e gerais | Específicas sobre amostragem, versões, XLA, memória e grafos |

## 4. Mudança do objetivo e das conclusões centrais

### 4.1 O que a versão antiga afirmava

A versão antiga dizia que o trabalho mostrava empiricamente uma ordem subótima
dos passes do PyTorch/TorchInductor e propunha uma sequência melhor. Essa era
uma afirmação causal e geral.

Em linguagem simples, ela dizia algo próximo de:

> “O compilador faz as otimizações na ordem errada.”

O experimento, porém, aplicava um preprocessamento fora do compilador e
comparava o resultado final. Ele não alterava isoladamente a ordem dos passes
internos do TorchInductor.

### 4.2 O que a nova versão afirma

A nova versão passa a afirmar que:

- o benchmark caracteriza o comportamento de pilhas nativas completas;
- o fold offline reduz o custo de compilação no caminho ResNet avaliado;
- o resultado sugere uma oportunidade de especialização de pesos congelados;
- não é possível concluir que a ordem interna dos passes seja incorreta;
- não existe evidência para uma ordem universal válida para qualquer modelo.

Essa mudança torna a conclusão compatível com o que foi realmente controlado.

### 4.3 Por que isso importa

Uma associação não é necessariamente uma causa.

```text
Fold antes da captura
        ↓
grafo menor
        ↓
compilação menor
```

Esse resultado mostra que o grafo menor está associado a uma compilação menor.
Para provar que “o pass A deve vir antes do pass B”, seria necessário modificar
somente essa ordem dentro do compilador e manter o restante constante.

## 5. Mudanças motivadas pelos revisores

Os revisores não pediram nominalmente “use Welch” ou “use Holm”. Eles pediram
correções mais gerais, e os autores escolheram essas técnicas para atendê-las.

### 5.1 Repetição das compilações

O Revisor 4 observou que o artigo repetia a execução 50 vezes, mas não
informava múltiplas compilações. Ele recomendou repetir a compilação e reportar
estatísticas.

Resposta na nova versão:

- \(K=5\) compilações frias por configuração ResNet;
- cada compilação ocorre em um processo independente;
- média, desvio-padrão amostral e IC de 95% no nível de processo;
- caches privados e persistência desativada.

### 5.2 Afirmação forte sobre a ordem dos passes

O Revisor 4 pediu cautela com a afirmação de que o fold deveria sempre vir
antes da fusion.

Resposta:

- a afirmação universal foi retirada;
- o resultado passou a ser descrito como oportunidade específica do caminho
  avaliado;
- o resumo e a conclusão agora dizem explicitamente que não foi estabelecida
  uma ordem universal.

### 5.3 Correção do modelo após o fold

O Revisor 4 perguntou se a acurácia ou outra medida de correção havia sido
avaliada.

Resposta:

- foram comparadas as saídas antes e depois do fold;
- usou-se `allclose(rtol=1e-3, atol=1e-4)`;
- registrou-se o erro absoluto máximo;
- confirmou-se que nenhuma BatchNorm permaneceu no modelo folded.

Esse teste não substitui uma avaliação completa de acurácia em um dataset, mas
verifica diretamente se a transformação preserva as saídas dentro da
tolerância escolhida.

### 5.4 Fundamentação teórica

O Revisor 4 apontou falta de referências na fundamentação teórica.

Resposta:

- foram adicionadas referências diretamente à definição de fusion;
- a seção de trabalhos relacionados foi movida para logo após a introdução;
- foram incluídos Optimus, DNNFusion e XNNC;
- legalidade e rentabilidade de uma transformação passaram a ser distinguidas.

### 5.5 Termos e figuras

Os revisores pediram:

- definição de kernels internos e externos;
- explicação de \(N,C,H,W\);
- definição de GEMM e QKV;
- legendas mais informativas;
- tradução dos textos que ainda estavam em português;
- referência explícita às figuras e tabelas.

A nova versão incorpora essas correções e reduz o uso de nomes internos de
arquivos e funções no corpo do artigo.

## 6. Novo protocolo experimental

### 6.1 Antes

O protocolo antigo informava:

- um processo isolado para compilação;
- 10 warmups;
- 50 execuções;
- média, desvio e intervalo de confiança calculáveis a partir das execuções.

O problema era que as 50 execuções compartilhavam uma única compilação. Elas
não mediam a variação que ocorreria se o modelo fosse compilado novamente.

### 6.2 Agora

Para cada célula ResNet definida por backend, modelo e input:

```text
processo frio 1 → compila → 10 warmups → 50 execuções → média 1
processo frio 2 → compila → 10 warmups → 50 execuções → média 2
processo frio 3 → compila → 10 warmups → 50 execuções → média 3
processo frio 4 → compila → 10 warmups → 50 execuções → média 4
processo frio 5 → compila → 10 warmups → 50 execuções → média 5
```

As cinco médias são as observações independentes usadas nos intervalos de
confiança e nos testes.

### 6.3 Experimental unit

A unidade experimental passou a ser o processo compilado independentemente.

Isso evita a **pseudorreplicação**, que ocorreria se as 250 execuções
\(5\times50\) fossem tratadas como 250 compilações diferentes.

As 50 execuções continuam úteis para medir o ruído dentro de cada processo e
permanecem armazenadas como dados brutos.

### 6.4 Isolamento de cache

A nova versão explicita que:

- TorchInductor, Triton e JAX recebem diretórios privados de cache;
- caches persistentes do TorchInductor e JAX são desativados;
- original e folded são compilados em processos diferentes;
- TVM e XLA não compilam primeiro uma variante auxiliar que possa aquecer o
  estado interno;
- arquivos de pesos e page cache do sistema operacional continuam
  compartilhados;
- model loading fica fora do intervalo cronometrado.

Isso aproxima o experimento de uma compilação fria real, sem afirmar que toda a
máquina foi reiniciada entre as observações.

## 7. Mudança na medição de compilação e execução

### 7.1 Compilation-cost proxy

Na versão antiga, os valores eram chamados simplesmente de tempo de
compilação. Na nova, são chamados de **proxy de custo de compilação específico
do backend**.

O motivo é que as APIs não oferecem a mesma fronteira de medição:

- TVM: cronometra passes do grafo e `relax.build`, depois da captura/importação
  FX;
- TorchInductor e XLA: usam
  \(T_C=T_{\text{first}}-T_X\), isto é, primeira chamada menos latência
  steady-state.

Assim, os números medem custos operacionais úteis, mas não exatamente o mesmo
conjunto de passes.

### 7.2 Steady-state latency

A execução passou a ser descrita explicitamente como latência sincronizada
depois dos warmups.

Os warmups permitem que inicialização do runtime e autotuning aconteçam antes
da janela principal. A sincronização impede que o cronômetro meça somente o
tempo de enfileirar trabalho assíncrono na GPU.

### 7.3 Exportação dos artefatos

HLO, TVMScript/TIR e wrappers Triton/Python são exportados fora da janela de
steady-state.

O custo de executar o wrapper faz parte da inferência. O que fica de fora é o
trabalho extra de salvar, ler e analisar arquivos para a pesquisa.

## 8. Estatística adicionada

### 8.1 Intervalos de confiança

Os intervalos de execução usam a distribuição t de Student sobre as cinco
médias de processo.

Em linguagem simples, a distribuição t reconhece que cinco processos ainda são
uma amostra pequena e, por isso, representa mais incerteza do que um intervalo
calculado como se houvesse centenas de repetições independentes.

### 8.2 Teste de Welch

Para cada input, Welch compara:

```text
5 médias de processos originais
contra
5 médias de processos folded
```

Ele pergunta se a distância entre as médias é grande em relação à variação dos
dois grupos. Foi escolhido porque original e folded são grupos independentes e
podem ter variâncias diferentes.

O teste é bilateral: tanto melhora quanto piora contam como diferença.

### 8.3 Correção de Holm

Foram feitas dez comparações por família: duas ResNets vezes cinco inputs.
Quando muitos testes são feitos, aumenta a chance de algum parecer positivo
apenas por sorte.

Holm torna o critério mais rigoroso levando em conta esse conjunto de testes.
Ele foi aplicado separadamente a:

- compilação depois do preprocessamento;
- custo fixo incluindo o preprocessamento;
- latência de execução.

### 8.4 Resultado estatístico

Na nova campanha:

- 9 de 10 reduções de compilação foram resolvidas após Holm;
- as mesmas 9 continuaram resolvidas depois de somar o tempo do fold;
- a exceção foi ResNet-50 em `(64,3,224,224)`, cuja amostra base teve alta
  variância;
- nenhuma diferença de execução da ResNet-18 foi resolvida;
- quatro speedups de execução da ResNet-50 foram resolvidos.

“Resolvida” significa que a diferença permaneceu estatisticamente detectável
depois da correção. Não significa que será reproduzida em qualquer máquina.

## 9. Comparação mais justa entre os backends

### 9.1 O que foi padronizado

A nova versão deixa claro que foram compartilhados:

- arquitetura nominal ResNet;
- shape;
- FP32;
- GPU;
- modo de inferência;
- cinco processos frios;
- 10 warmups;
- 50 execuções sincronizadas;
- formato de exportação das medições.

### 9.2 O que não foi igualado

Cada backend manteve sua rota nativa:

- TorchInductor e TVM usam definições `torchvision`;
- essas duas rotas usam `weights=None` e seed 0;
- original e folded do TorchInductor começam de `state_dicts` idênticos;
- TVM importa o grafo FX e vincula parâmetros;
- XLA usa `flaxmodels`;
- XLA usa `pretrained=None`, sem normalização de entrada embutida, e retorna
  logits;
- TorchInductor usa NCHW lógico em memória `channels_last`;
- TVM usa NCHW;
- XLA usa NHWC;
- parâmetros JAX ficam fechados na função JIT;
- TorchInductor mantém um contrato mais dinâmico para os parâmetros do módulo.

Os inputs também não são numericamente iguais: TorchInductor/TVM usam dados
aleatórios com seed, enquanto XLA usa valores um.

Fixar a seed torna uma rota reproduzível, mas não torna os pesos de PyTorch e
JAX numericamente iguais, pois as bibliotecas podem usar inicializadores e
ordens de geração diferentes.

### 9.3 Nova interpretação de fairness

A versão antiga dizia que os compiladores estavam configurados de forma
equivalente. A versão nova substitui essa formulação por uma descrição mais
precisa:

> As condições externas são comuns, mas cada pilha mantém seu frontend, layout,
> contrato de parâmetros e API de compilação.

Consequentemente, o benchmark responde:

> “Qual pilha nativa foi mais adequada neste ambiente?”

Ele não responde isoladamente:

> “Qual compilador produz o melhor código quando recebe exatamente o mesmo
> programa?”

Também não foi estabelecida equivalência numérica dos modelos entre os três
backends. A verificação `allclose` foi feita entre original e folded dentro do
experimento ResNet do TorchInductor, não entre TorchInductor, TVM e XLA.

## 10. Mudança na interpretação de kernels e fusion

### 10.1 Antes

As figuras eram apresentadas como contagens ou taxas de kernels e permitiam uma
leitura próxima de um ranking de fusion.

### 10.2 Agora

As figuras são descritas como um **censo estático de unidades semelhantes a
chamadas**:

- TVM: chamadas/funções TIR;
- TorchInductor: chamadas no wrapper para Triton ou bibliotecas externas;
- XLA: HLO fusions e custom calls.

Essas unidades estão em níveis de abstração diferentes. Uma HLO fusion, uma
função TIR e uma chamada Python não representam necessariamente um único
lançamento de GPU.

### 10.3 Consequência

As figuras agora servem para explicar a estrutura interna:

- TVM gera mais unidades internamente;
- TorchInductor combina Triton com chamadas externas;
- XLA combina HLO fusions e custom calls.

Elas não são usadas para afirmar que “menos chamadas significa compilador
melhor”.

Uma comparação normalizada exigiria:

- o mesmo grafo inicial;
- o mesmo contrato de parâmetros;
- o mesmo nível de contagem;
- idealmente profiling dinâmico dos lançamentos de GPU.

## 11. Novos resultados das ResNets

As tabelas antigas de ResNet foram substituídas por tabelas geradas a partir da
campanha repetida. Agora cada célula mostra média e desvio-padrão amostral sobre
cinco médias de processo.

Também mudou o uso de cores:

- antes, melhores e piores valores pontuais recebiam destaque;
- agora, o menor valor recebe destaque apenas quando seu IC marginal de 95% não
  se sobrepõe aos dois concorrentes;
- o destaque vermelho de “pior” foi removido.

A interpretação estrutural também foi suavizada:

- características do TVM são atribuídas ao caminho de importação e lowering
  avaliado, não a uma limitação universal do TVM;
- o maior código da ResNet-50 no TorchInductor é associado ao maior custo de
  compilação, sem ser tratado como causa isolada;
- a estrutura das custom calls do XLA é descrita, mas a diferença de tempo não
  é atribuída apenas à fusion.

## 12. Fold de Conv–BatchNorm nas ResNets

### 12.1 Transformação corretamente identificada

A versão antiga aproximava o fold de uma fusion
`Conv+BatchNorm+ReLU` em um único kernel.

A versão nova separa duas operações:

1. o fold absorve os parâmetros da BatchNorm na convolução antes da captura;
2. o backend pode ou não incorporar a ReLU como epílogo da convolução.

O experimento garante somente o primeiro item. Ele não garante um único kernel
em runtime.

### 12.2 Custo do preprocessamento

O tempo do fold passou a ser medido:

- ResNet-18: aproximadamente 105–111 ms;
- ResNet-50: aproximadamente 118–124 ms.

As tabelas mostram o custo de compilação após o fold. O deployment envelope
soma:

\[
T_{\text{fold}}+T_{\text{compilação após fold}}
\]

Isso evita oferecer o preprocessamento gratuitamente à variante folded.

### 12.3 Correção numérica

As saídas passaram em:

```text
allclose(rtol=1e-3, atol=1e-4)
```

Erros absolutos máximos:

- ResNet-18: \(6{,}676\times10^{-6}\);
- ResNet-50: \(2{,}441\times10^{-4}\).

Também foi confirmado que nenhuma BatchNorm permaneceu após a transformação.

### 12.4 Novas magnitudes

As reduções pontuais antigas eram:

- ResNet-18: aproximadamente 49,6%–58,9%;
- ResNet-50: aproximadamente 17,5%–43,6%.

Na campanha repetida, passaram a:

- ResNet-18: 29,6%–40,3%;
- ResNet-50: 14,0%–27,5%.

Os números novos são menos espetaculares, mas têm suporte experimental mais
forte.

### 12.5 Mecanismo sustentado pelos artefatos

A nova versão não afirma que o ganho veio necessariamente de menos lançamentos.
Os wrappers mostram que muitas contagens permaneceram iguais, enquanto o
tamanho do código gerado caiu:

- ResNet-18: redução de 38%–44% no código;
- ResNet-50: redução de aproximadamente 37%.

A conclusão sustentada é:

> O fold reduziu o volume de codegen.

Para concluir que reduziu lançamentos reais seria necessário usar um profiler
dinâmico.

## 13. Retirada do resultado de fold do BERT

Esta é uma das mudanças mais importantes.

### 13.1 O resultado antigo

A versão antiga relatava reduções de compilação de 46,3%–52,8% ao supostamente
absorver a parte afim da LayerNorm na Linear.

### 13.2 O problema encontrado

A auditoria mostrou que:

- o matcher procurava `LayerNorm` e `Linear` adjacentes como siblings;
- ele encontrou zero pares;
- portanto, fez zero rewrites;
- base e “folded” eram o mesmo grafo;
- as duas versões eram compiladas sequencialmente no mesmo processo;
- a segunda compilação podia reutilizar estado interno já aquecido.

Assim, a redução antiga não era efeito de um fold real.

### 13.3 Problema algébrico no BERT usado

O BERT avaliado é post-LayerNorm. A saída da LayerNorm também alimenta uma
conexão residual.

```text
saída da LayerNorm
├── projeções Linear
└── caminho residual
```

Absorver \(\gamma,\beta\) somente nas projeções mudaria o valor do caminho
residual e, consequentemente, a função do modelo.

### 13.4 Experimento corrigido

Foi criado um controle que:

- faz uma cópia profunda do mesmo grafo;
- executa zero rewrites;
- compila base e controle em processos frios separados;
- usa cinco processos por variante;
- mantém seeds e protocolo de entrada;
- compara censo estrutural e tamanho de código.

Os intervalos de confiança se sobrepõem em todas as comparações. O resultado
antigo foi formalmente retirado.

### 13.5 Nova conclusão

A derivação `LayerNorm→Linear` continua matematicamente válida quando todos os
consumidores podem ser compensados. Ela não é apresentada como uma otimização
válida para o BERT concreto usado no artigo.

## 14. BERT e GPT-2 agora são históricos e descritivos

As tabelas cross-backend de BERT e GPT-2 foram preservadas, mas agora são
rotuladas como históricas.

Elas possuem:

- uma única compilação por célula;
- 50 execuções que compartilham aquele processo;
- desvio dentro do processo, não entre compilações independentes.

Por isso:

- não há vencedor estatístico declarado;
- os números servem para gerar hipóteses;
- não sustentam rankings inferenciais entre backends;
- não são misturados à nova campanha ResNet \(K=5\).

### 14.1 Auditoria estrutural do BERT

Uma recaptura atual do TorchInductor preservou as 72 multiplicações de
matrizes, mas mudou o grafo geral:

- grafo histórico: base de 500 operações;
- grafo atual: 465 nós FX;
- atenção passou a usar um operador dedicado;
- o tratamento de bias gerou chamadas Triton adicionais.

Por isso a taxa de redução histórica não foi silenciosamente substituída. Uma
comparação atual exigiria recapturar TVM, TorchInductor e XLA juntos.

## 15. Correção do modelo de crossover

### 15.1 Problema da formulação antiga

A versão antiga usava:

\[
n_{\mathrm{eq}}=\max\left(0,\frac{b_b-b_f}{a_f-a_b}\right)
\]

Transformar todo resultado negativo em zero misturava situações diferentes:

- fold domina desde o início;
- não existe cruzamento positivo;
- as retas cruzariam apenas em um \(n\) sem sentido físico.

### 15.2 Formulação nova

A nova versão calcula:

\[
n_{\mathrm{cross}}=\frac{b_b-b_f}{a_f-a_b}
\]

e interpreta explicitamente os sinais.

Quando o fold possui menor custo fixo e latência não maior, ele domina para
todo \(n\geq0\). O zero da figura é apenas um marcador para “não há crossover;
fold domina”.

### 15.3 Único crossover finito do fold

Na ResNet-18 com `(16,3,224,224)`, o ponto estimado é aproximadamente 186.206
execuções:

- folded é preferível antes desse valor;
- original é preferível depois;
- isso ocorre porque a latência folded foi 0,3% maior;
- essa pequena diferença de execução não foi resolvida após Holm.

Portanto, o ponto é descritivo, não uma fronteira estatisticamente estabelecida.

## 16. Mudança no benchmark e no deployment envelope

O benchmark continua usando:

\[
T_b(n)=T_C+nT_X
\]

Mas a interpretação mudou:

- a linha representa uma pilha nativa medida;
- o intercepto é um proxy de custo fixo;
- a inclinação é a latência steady-state;
- o envelope usa estimativas pontuais;
- a recomendação depende de máquina, versões, flags, layout e contrato do
  modelo;
- o envelope não é um ranking com incerteza estatística incorporada.

O exemplo da figura também mudou com os novos dados:

- versão antiga: folded TorchInductor até aproximadamente 5.197 execuções e
  XLA depois;
- versão nova: XLA até aproximadamente 44.174 execuções e folded TorchInductor
  depois.

Na nova figura, o intercepto folded inclui o tempo medido do preprocessamento.

## 17. Ambiente experimental atualizado

A nova versão separa campanhas que antes pareciam pertencer ao mesmo ambiente.

### 17.1 ResNets repetidas

- Fedora 41;
- kernel 6.16.12;
- Ryzen 5 4600G;
- 15,4 GiB de RAM;
- RTX 3050 de 8 GB;
- driver 580.105.08;
- `nvcc` CUDA 12.6;
- TorchInductor/PyTorch 2.9.0 com CUDA 12.8 empacotado;
- JAX/jaxlib 0.6.2;
- TVM 0.22.dev0.

### 17.2 Transformers históricos

Continuam associados à campanha anterior:

- driver 580.95.05;
- TorchInductor/PyTorch 2.5.1;
- protocolo de processo único.

### 17.3 Auditoria BERT

- Python 3.11.13;
- PyTorch 2.5.1+cu121;
- Transformers 4.57.0.

### 17.4 Controle da cuDNN no XLA

A cuDNN empacotada produziu falhas intermitentes no planejamento de
convoluções. A campanha repetida passou a carregar a cuDNN 9.11.0 do host,
registrar versões de build e runtime e guardar falhas apenas como metadados de
retry.

Tentativas que falharam não entram como observações de tempo.

## 18. Limitações agora explicitadas

A nova versão registra que:

- cinco processos ainda são poucos;
- todos os resultados vêm de uma única máquina;
- BERT/GPT-2 cross-backend continuam sem repetição independente;
- a auditoria BERT não substitui essas tabelas históricas;
- inputs de \(1024^2\) podem limitar planos XLA por falta de memória;
- versões de framework mudam tempos e grafos;
- layouts e parâmetros diferem entre backends;
- o modelo linear ignora recompilações, caches e comportamentos não lineares;
- as recomendações não devem ser extrapoladas automaticamente para outro
  hardware ou software.

Essas limitações não anulam o estudo. Elas delimitam exatamente onde as
conclusões são válidas.

## 19. Mudanças no apêndice matemático

### 19.1 Nome e escopo

O apêndice deixou de chamar todas as três expressões de “fusions”. Agora
distingue:

- uma expressão de kernel fusion: `BatchNorm+ReLU`;
- um fold condicional de parâmetros: `Conv–BatchNorm`;
- um fold condicional: parte afim de `LayerNorm–Linear`.

### 19.2 Conv–BatchNorm–ReLU

A versão antiga dizia que a sequência resultava em um único kernel.

A nova versão diz:

- o benchmark garante o fold de BatchNorm nos parâmetros da convolução;
- a ReLU pode ser implementada como epílogo;
- transformar tudo em um único kernel é uma decisão do backend.

### 19.3 LayerNorm–Linear

A versão antiga descrevia a absorção de \(\gamma,\beta\) como aplicável ao BERT.

A nova versão adiciona a condição:

> Todo uso da saída afim da LayerNorm precisa ser removido ou compensado.

Como o BERT usado possui consumidor residual, a equação é uma derivação
condicional, não um resultado experimental válido para esse modelo.

## 20. Mudanças de escrita, estrutura e apresentação

Também ocorreram mudanças menores, mas úteis:

- `Related Work` foi movida para depois da introdução;
- o resumo agora apresenta protocolo, resultados numéricos e limitações;
- “CFG” foi substituído, onde apropriado, por grafo computacional, grafo de
  dados ou IR;
- “input size \(n\)” foi corrigido para “número de execuções \(n\)”;
- as figuras de fusion agora se concentram nas ResNets e no input
  `(1,3,224,224)`;
- legendas passaram a dizer que as contagens são estáticas;
- tabelas ResNet e fold passaram a ser geradas por script a partir dos JSONs;
- a macro `\tcode` tornou-se robusta para uso em títulos e legendas;
- o destaque vermelho de pior resultado foi removido;
- os metadados da conferência foram marcados como verificados;
- as conclusões causais foram substituídas por linguagem de associação e
  condicionamento.

## 21. Mudanças correspondentes no artefato e nos scripts

Para sustentar a nova versão, o repositório passou a incluir ou usar:

- repetição com `--compile-repeats 5`;
- processos independentes para cada variante;
- caches privados por processo;
- desativação de caches persistentes;
- armazenamento das médias de execução de cada processo;
- armazenamento das 250 amostras brutas por célula;
- IC de 95% no nível de processo;
- script de Welch + Holm:
  [`analyze_fold_stats.py`](../../scripts/analyze_fold_stats.py);
- auditoria de aplicabilidade do fold no BERT:
  [`results/bert_fold/README.md`](../../results/bert_fold/README.md);
- auditoria das contagens históricas de Transformers:
  [`results/transformer_ir/README.md`](../../results/transformer_ir/README.md);
- geração automática das tabelas em
  [`generated/`](generated/);
- registro de retries e versões da cuDNN;
- medição do tempo de preprocessamento do fold;
- registro do tamanho do código e do censo estrutural dos artefatos.

## 22. O que permaneceu

Nem tudo mudou. Permaneceram:

- o tema central: trade-off entre compilação e execução;
- os backends TVM, XLA e TorchInductor;
- as arquiteturas ResNet-18, ResNet-50, BERT e GPT-2;
- os cinco formatos de entrada das ResNets;
- 10 warmups e 50 execuções cronometradas;
- o modelo de custo \(T(n)=T_C+nT_X\);
- a análise de HLO, TVMScript/TIR e Triton;
- a ideia de usar o envelope inferior para orientar regimes de implantação;
- a derivação algébrica do fold Conv–BatchNorm em inferência.

O que mudou foi principalmente a força estatística, a definição do que está
sendo comparado e o cuidado com as conclusões.

## 23. Glossário simples

**Backend:** parte do sistema que recebe um modelo/grafo e produz código para o
hardware.

**Cold process:** processo novo, sem reutilizar o estado de uma compilação
anterior.

**Codegen:** geração do código final que será executado.

**Compilation-cost proxy:** aproximação operacional do custo de compilação
quando a API não expõe um cronômetro puro para todos os seus passos.

**Confidence interval:** faixa de valores plausíveis para a média, considerando
a variação observada.

**Custom call:** operação em HLO que delega trabalho para uma biblioteca
especializada, como cuDNN.

**Deployment envelope:** conjunto das linhas \(T(n)\); para cada \(n\), escolhe
a linha com menor custo estimado.

**Fold:** alteração antecipada dos parâmetros para remover uma operação do
grafo sem mudar a função matemática pretendida.

**Fusion:** combinação de operações durante a compilação para executá-las de
forma conjunta.

**Frontend graph:** representação inicial do modelo entregue ao compilador.

**Holm:** correção que reduz a chance de falso positivo quando muitos testes
são realizados.

**HLO/TIR:** representações intermediárias usadas respectivamente por XLA e
TVM.

**Parameter binding:** modo como os pesos reais são associados aos parâmetros
simbólicos do grafo.

**Pseudorreplicação:** tratar medições que compartilham o mesmo processo ou a
mesma compilação como experimentos completamente independentes.

**Steady-state latency:** tempo de execução depois de compilação, warmup e
inicializações.

**Welch:** teste que compara duas médias levando em conta o tamanho da amostra e
a variabilidade de cada grupo.

## 24. Conclusão

A versão nova é metodologicamente mais conservadora e mais defensável.

Ela troca uma narrativa forte — “encontramos uma ordem melhor de passes e
determinados backends fundem melhor” — por uma conclusão apoiada diretamente
pelos dados:

> Em cinco processos frios por configuração, o fold offline e numericamente
> validado de Conv–BatchNorm reduziu o custo fixo do TorchInductor em nove de
> dez inputs após correção estatística. Os efeitos de execução foram menores.
> A comparação cross-backend descreve pilhas nativas completas, e não isola a
> qualidade da fusion nem estabelece uma ordem universal de passes.

Essa conclusão é menos abrangente, porém estatisticamente e metodologicamente
mais justa.
