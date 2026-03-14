from scipy import stats
from mae import mae
from mae_log import mae_log


def calculate_metrics(true_density, estimated_density):
  mae_value = mae(true_density, estimated_density)
  mae_log_value = mae_log(true_density, estimated_density)
  pearson_value = stats.pearsonr(true_density, estimated_density)
  spearman_value = stats.spearmanr(true_density, estimated_density)
  print("Mae:", mae_value)
  print("Log mae:", mae_log_value)
  print("Pearson:", pearson_value)
  print("Spearman:", spearman_value)
  return {"mae_value": mae_value, "mae_log_value":mae_log_value, "pearson_value":pearson_value[0], \
        "spearman_value":spearman_value[0]}


