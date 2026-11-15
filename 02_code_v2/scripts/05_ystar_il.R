# IL-BMP reference timing: day at which the 1% net 3 d criterion is met (summBg),
# computed per submitted file from cumulative CH4 and blank bottles.
# Run only after the IIS models are locked:  Rscript 02_code_v2/scripts/05_ystar_il.R
suppressMessages(library(biogas))
root <- Sys.getenv("BMP_ROOT", ".")
raw <- file.path(root, "01_data_v2/raw/IL-BMP_Hafner2026")
out <- file.path(root, "03_outputs_v2/il"); dir.create(out, showWarnings = FALSE, recursive = TRUE)

setup <- read.csv(file.path(raw, "setup.csv"))
gas <- read.csv(file.path(raw, "biogas.csv"))
setup$m_sub_vs <- setup$mass_sub * setup$c_vs_sub_m
nom <- c(1, 2, 4)
isr_nom <- function(x) {
  y <- ifelse(is.finite(x) & x > 0, x, 2)
  nom[apply(abs(outer(log(y), log(nom), "-")), 1, which.min)]
}
setup$cond <- ifelse(setup$substrate == "Blank", "Blank",
                     paste(setup$substrate, isr_nom(setup$ISR), sep = "|"))
res <- list()
for (fid in unique(setup$file_id)) {
  st <- setup[setup$file_id == fid, ]
  if (!any(st$substrate == "Blank")) next
  vl <- gas[gas$file_id == fid, c("bottle_id", "time_d", "cvCH4")]
  vl <- vl[order(vl$bottle_id, vl$time_d), ]
  r <- tryCatch(
    summBg(vol = vl, setup = st, id.name = "bottle_id", time.name = "time_d",
           descrip.name = "cond", inoc.name = "Blank", inoc.m.name = "mass_inoc",
           norm.name = "m_sub_vs", vol.name = "cvCH4", when = "1p3d",
           show.obs = TRUE, quiet = TRUE),
    error = function(e) { message("file ", fid, ": ", conditionMessage(e)); NULL })
  if (!is.null(r)) { r$file_id <- fid; res[[length(res) + 1]] <- r }
}
obs <- do.call(rbind, res)
write.csv(obs, file.path(out, "il_tstar_biogas_bottles.csv"), row.names = FALSE)
cat("bottle rows:", nrow(obs), "\n")
