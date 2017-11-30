from src.modules.hermes import Braco
import numpy as np
from time import sleep
b = Braco(autocommit=True)
b._conectado = True
b.commit()

sleep(4)

A, B = list(), list()
A.append(b.coeficientes_cinematica_direta)
B.append(b.posicao[:-1:])

b.servos[0].angulo = 30
b.servos[1].angulo = 135

sleep(4)

A.append(b.coeficientes_cinematica_direta)
B.append(b.posicao[:-1:])

b.servos[0].angulo = 60
b.servos[1].angulo = 0

sleep(4)

A.append(b.coeficientes_cinematica_direta)
B.append(b.posicao[:-1:])

b.servos[0].angulo = 60
b.servos[1].angulo = 90
b.servos[2].angulo = -30
b.servos[3].angulo = -30

sleep(4)

A.append(b.coeficientes_cinematica_direta)
B.append(b.posicao[:-1:])

b.encontrar_constantes(np.array(A), np.array(B))
