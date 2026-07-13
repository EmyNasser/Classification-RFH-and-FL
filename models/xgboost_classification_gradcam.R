# =========================================================
#  XGBoost Pipeline: Train / Validation / External Test
#  Multiclass Model Using CNN Embeddings (AlexNet/ResNet18/VGG16)
#  Unified Metric Plot (Train -> Validation -> External Test)
# =========================================================
#
# This is the multiclass counterpart of the original binary "Nucleus features"
# XGBoost script. Instead of tabular nuclear morphometric features (Area,
# Perimeter, Circularity...), it consumes the embedding CSVs produced by
# extract_embeddings.py:
#
#     embeddings_train.csv
#     embeddings_val.csv     (optional -- auto-split from train if missing)
#     embeddings_test.csv
#
# Each CSV has columns: class_name, label, feat_0000, feat_0001, ...
# (label is 0-indexed, exactly as saved by class_map.csv during CNN training)
#
# Works for any number of classes (e.g. num_class = 21 for UCMerced), not
# just binary RFH/FL.

# --- Load Required Libraries ---
packages <- c(
  "xgboost", "caret", "pROC", "MLmetrics", "dplyr",
  "ggplot2", "readxl", "scales", "reshape2"
)

for (pkg in packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, repos = "https://cran.r-project.org/")
    library(pkg, character.only = TRUE)
  }
}

# =========================================================
#  ---- Config (edit these paths) ----
# =========================================================
# Point these at the output of extract_embeddings.py, e.g.:
#   python extract_embeddings.py --backbone resnet18 --results_dir /kaggle/working/results_resnet18
# writes to /kaggle/working/results_resnet18/embeddings/embeddings_{train,val,test}.csv
embeddings_dir  <- "/kaggle/working/results/embeddings"
train_csv_path  <- file.path(embeddings_dir, "embeddings_train.csv")
val_csv_path    <- file.path(embeddings_dir, "embeddings_val.csv")   # set to NA to auto-split from train instead
test_csv_path   <- file.path(embeddings_dir, "embeddings_test.csv")

output_dir      <- "Results_XGBoost_Embeddings"
val_split_frac  <- 0.8   # only used if val_csv_path is NA / missing (train-side 80/20 split)
random_seed     <- 123

dir.create(output_dir, showWarnings = FALSE)

# =========================================================
#  Load Datasets
# =========================================================
read_embeddings <- function(path) {
  if (is.na(path) || !file.exists(path)) return(NULL)
  df <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  feat_cols <- grep("^feat_", colnames(df), value = TRUE)
  if (length(feat_cols) == 0) {
    stop(sprintf("No 'feat_XXXX' columns found in '%s'. Check the file format.", path))
  }
  list(
    X = as.matrix(df[, feat_cols, drop = FALSE]),
    y = as.integer(df$label),
    class_name = df$class_name
  )
}

train_full <- read_embeddings(train_csv_path)
if (is.null(train_full)) {
  stop(sprintf("Training embeddings not found at '%s'. Run extract_embeddings.py first.", train_csv_path))
}
val_raw  <- read_embeddings(val_csv_path)
test_raw <- read_embeddings(test_csv_path)
if (is.null(test_raw)) {
  stop(sprintf("Test embeddings not found at '%s'. Run extract_embeddings.py first.", test_csv_path))
}

num_class <- length(unique(train_full$y))
cat(sprintf("Detected %d classes.\n", num_class))

# =========================================================
#  Split Dataset (only if a val CSV wasn't provided)
# =========================================================
set.seed(random_seed)

if (is.null(val_raw)) {
  cat("[INFO] No validation CSV provided -- auto-splitting the training set "
      , sprintf("(%.0f%% train / %.0f%% val).\n", val_split_frac * 100, (1 - val_split_frac) * 100))
  train_index <- createDataPartition(train_full$y, p = val_split_frac, list = FALSE)

  X_train <- train_full$X[train_index, , drop = FALSE]
  y_train <- train_full$y[train_index]

  X_val <- train_full$X[-train_index, , drop = FALSE]
  y_val <- train_full$y[-train_index]
} else {
  X_train <- train_full$X
  y_train <- train_full$y
  X_val   <- val_raw$X
  y_val   <- val_raw$y
}

X_test <- test_raw$X
y_test <- test_raw$y

dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dval   <- xgb.DMatrix(data = X_val,   label = y_val)
dtest  <- xgb.DMatrix(data = X_test,  label = y_test)

watchlist <- list(train = dtrain, val = dval)

# =========================================================
#  Train XGBoost Model (multiclass)
# =========================================================
params <- list(
  objective = "multi:softprob",
  num_class = num_class,
  booster = "gbtree",
  eta = 0.01,
  max_depth = 8,
  subsample = 0.9,
  colsample_bytree = 0.9,
  eval_metric = "mlogloss"
)

set.seed(random_seed)
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 3000,
  watchlist = watchlist,
  early_stopping_rounds = 100,
  verbose = 1
)

xgb.save(xgb_model, file.path(output_dir, "xgb_model_multiclass.model"))

# =========================================================
#  Evaluation: Train / Validation / External Test (multiclass)
# =========================================================
get_metrics_multiclass <- function(y_true, prob_matrix, num_class) {
  # prob_matrix: N x num_class matrix of class probabilities
  pred_labels <- max.col(prob_matrix) - 1   # 0-indexed, matches y_true encoding

  y_true_f <- factor(y_true, levels = 0:(num_class - 1))
  pred_f   <- factor(pred_labels, levels = 0:(num_class - 1))

  conf_mat <- confusionMatrix(pred_f, y_true_f)

  # Macro-averaged Precision / Recall / F1 across all classes
  precisions <- sapply(0:(num_class - 1), function(cls) {
    Precision(y_pred = pred_labels, y_true = y_true, positive = cls)
  })
  recalls <- sapply(0:(num_class - 1), function(cls) {
    Recall(y_pred = pred_labels, y_true = y_true, positive = cls)
  })
  f1s <- sapply(0:(num_class - 1), function(cls) {
    F1_Score(y_pred = pred_labels, y_true = y_true, positive = cls)
  })

  # Multiclass AUC (Hand & Till, one-vs-one average) via pROC
  auc_value <- tryCatch({
    as.numeric(multiclass.roc(y_true, prob_matrix)$auc)
  }, error = function(e) NA)

  data.frame(
    Accuracy  = conf_mat$overall["Accuracy"],
    Precision = mean(precisions, na.rm = TRUE),
    Recall    = mean(recalls, na.rm = TRUE),
    F1_Score  = mean(f1s, na.rm = TRUE),
    AUC       = auc_value
  )
}

predict_probs <- function(model, dmat, num_class) {
  raw <- predict(model, dmat)
  matrix(raw, ncol = num_class, byrow = TRUE)
}

train_probs <- predict_probs(xgb_model, dtrain, num_class)
val_probs   <- predict_probs(xgb_model, dval,   num_class)
test_probs  <- predict_probs(xgb_model, dtest,  num_class)

train_metrics <- get_metrics_multiclass(y_train, train_probs, num_class)
val_metrics   <- get_metrics_multiclass(y_val,   val_probs,   num_class)
test_metrics  <- get_metrics_multiclass(y_test,  test_probs,  num_class)

all_metrics <- rbind(
  cbind(Set = "Training", train_metrics),
  cbind(Set = "Validation", val_metrics),
  cbind(Set = "External Test", test_metrics)
)

write.csv(all_metrics, file.path(output_dir, "all_metrics_multiclass.csv"), row.names = FALSE)

# =========================================================
#  Visualization
#  Ordered: Training -> Validation -> External Test
# =========================================================
all_long <- reshape2::melt(all_metrics, id.vars = "Set",
                           variable.name = "Metric", value.name = "Value")

all_long$Set <- factor(all_long$Set,
                       levels = c("Training", "Validation", "External Test"))

p <- ggplot(all_long, aes(x = Metric, y = Value, fill = Set)) +
  geom_bar(stat = "identity",
           position = position_dodge(width = 0.8),
           width = 0.6) +
  geom_text(aes(label = sprintf("%.3f", Value)),
            position = position_dodge(0.8),
            vjust = -0.5,
            size = 3.5) +
  scale_fill_manual(values = c("#2166AC", "#67A9CF", "#D1E5F0")) +
  ylim(0, 1.05) +
  labs(
    title = sprintf("XGBoost Performance (CNN Embeddings, %d classes)", num_class),
    subtitle = "Training | Validation | External Test",
    x = "Metric",
    y = "Value",
    fill = "Dataset"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "top",
    legend.title = element_text(face = "bold")
  )

print(p)

ggsave(
  file.path(output_dir, "all_datasets_metrics_multiclass.png"),
  p, width = 9, height = 5.5, dpi = 300
)

cat(sprintf("\nDone. Results saved to '%s'.\n", output_dir))
