"""MLflow Model Registry: mantém um alias apontando para a melhor versão do modelo."""

from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException


def get_version_by_alias(client: MlflowClient, model_name: str, alias: str) -> ModelVersion | None:
    """Versão apontada pelo alias, ou ``None`` se o alias ainda não existe."""
    try:
        return client.get_model_version_by_alias(model_name, alias)
    except MlflowException:
        return None


def get_version_metric(client: MlflowClient, version: ModelVersion, metric: str) -> float:
    """Valor da métrica na run que gerou a versão do modelo."""
    return float(client.get_run(version.run_id).data.metrics[metric])


def promote_if_best(
    client: MlflowClient, model_name: str, version: str | int, metric: str, alias: str
) -> bool:
    """Aponta ``alias`` para ``version`` se ela superar a versão campeã atual.

    Maior valor de ``metric`` é melhor. Em caso de empate, a campeã atual é mantida.
    Retorna ``True`` se a versão foi promovida.
    """
    candidate = client.get_model_version(model_name, str(version))
    champion = get_version_by_alias(client, model_name, alias)

    if champion is not None and str(champion.version) != str(candidate.version):
        candidate_score = get_version_metric(client, candidate, metric)
        champion_score = get_version_metric(client, champion, metric)
        if candidate_score <= champion_score:
            print(f"Versão {candidate.version} ({metric}={candidate_score:.4f}) não supera a "
                  f"'{alias}' v{champion.version} ({metric}={champion_score:.4f}).")
            return False

    client.set_registered_model_alias(model_name, alias, str(candidate.version))
    print(f"Versão {candidate.version} de '{model_name}' promovida a '{alias}'.")
    return True
