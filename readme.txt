
                                            MULTI TRAFFIC SIGN CLASSIFICATION

Este trabalho apresenta 4 estratégias de treinamento de modelo CNN para classificação de sinais de trânsito de veículos autônomos quando circulam entre países. 

Os programas em python tiveram com referência https://github.com/chsasank/Traffic-Sign-Classification.keras e foram feitas adaptações.

Os conjuntos de dados públicos de treino e teste dos países estão em:
Alemão: http://benchmark.ini.rub.de/?section=gtsrbsubsection=dataset2 
Belga:  https://btsd.ethz.ch/shareddata/ 
Croata: http://www.zemris.fer.hr/~kalfa/Datasets/rMASTIF/

Foram criados os diretórios "AlemaoTraf","BelgaTraf" e "CroataTraf" e abaixo de cada um destes diretorios foram
criados os subdiretorios "Training" e "Testing" onde foram feitos os downloads dos datasets de cada país.


1) Treinamento individual

O treinamento individual foi feito entrando em cada diretorio de cada país usando a seguinte sintaxe:  
python T_pais.py <pais.csv> <num_classes> <batch_size>
"pais.csv" corresponde ao arquivo csv que contem como imagens e suas classes correspondentes ao conjunto de dados de teste deste país.
        Alemão=Alemao.csv, Belga=Belga.csv, Croata=Croata.csv        
 num_classes corresponde ao número de classes deste país. (Alemão=43, Belga=62, Croata=31)
 batch_size corresponde ao tamanho do lote que deseja consultar.

Exemplo: para o treino do conjunto de dados alemão que contem 43 classes e deseja executar com lote = 48.
python T_pais.py Alemao.csv 43 48

2) Treinamento combinado simples


3) Treinamento combinado completo


4) Treinamento com dataset único.

Para fazer este treinamento foi criado um dataset unico a partir do script gera_dataset_unico.sh que faz o seguinte:

1) Copia os datasets AlemaoTraf(treino),BelgaTraf(treino,teste) e CroataTraf(treino,teste) para um diretorio chamado alemaobelgacroata.
2) Renomeia os arquivos  dos datasets belga concatenando a letra "b" em todos os nomes dos arquivos belgas e "c" em todos os nomes dos arquivos croata.Isto é feito pois existe nomes iguais.
3) Cria-se diretorios correspondentes as classes que nao tem equivalencia de classes entre os datasets.
4) Copia-se os arquivos dos diretorio belga e croata para o diretorio alemao de classe equivalente.
5) Cria os arquivos csv belga e croata. O csv alemão ja existente se mantem.
6) Copia o datatase AlemaoTraf(teste) para alemaobelgacroata\alemao\teste

Para executar o treinamento a sintaxe de para cada país é:
cd AlemaoBelgaCroata/Alemao
Alemao : python T_AlBeCr.py   Alemao.csv  88 <batch_size>
Belga:   python T_AlBeCr.py   BelgaB.csv  88 <batch_size>
Croata:  python T_AlBeCr.py   CroataC.csv 88 <batch_size>


--------------------------------------FIM GIT---------------------------

