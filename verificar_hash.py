import importlib.metadata as metadata
import hashlib
import base64

dist = metadata.distribution("scikit-learn")

for arquivo in dist.files or []:
    if arquivo.name == "_argkmin.cp314-win_amd64.pyd":
        caminho = dist.locate_file(arquivo)
        esperado = arquivo.hash.value if arquivo.hash else None

        calculado = base64.urlsafe_b64encode(
            hashlib.sha256(caminho.read_bytes()).digest()
        ).decode().rstrip("=")

        print("Arquivo:", caminho)
        print("Hash esperado:", esperado)
        print("Hash calculado:", calculado)
        print("Coincidem:", calculado == esperado if esperado else "NAO VERIFICADO")
        break
else:
    print("Arquivo nao encontrado nos registros da instalacao.")