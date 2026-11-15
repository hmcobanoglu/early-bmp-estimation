# Reference BMP (Y*) for IIS-BMP with the biogas package (same software family the
# data owners used), 1% net 3 d criterion. Output compared with Hafner et al. 2020
# supplementary subset D (BMP_1p3dnet).
# Run: Rscript 02_code_v2/scripts/02_ystar_biogas_iis.R
suppressMessages(library(biogas))
root <- Sys.getenv("BMP_ROOT", ".")
raw <- file.path(root, "01_data_v2/raw/IIS-BMP_Hafner2022")
out <- file.path(root, "03_outputs_v2/inventory")

bottles <- read.csv(file.path(raw, "bottles.csv"))
nom <- c(1, 2, 4)
isr_nom <- function(x) {
  y <- ifelse(is.finite(x) & x > 0, x, 2)
  nom[apply(abs(outer(log(y), log(nom), "-")), 1, which.min)]
}
bottles$cond <- ifelse(bottles$substrate == "BK", "BK",
                       paste(bottles$substrate, isr_nom(bottles$ISR), sep = "|"))
gas <- read.csv(file.path(raw, "biogas.csv"))
gas <- merge(gas, bottles[, c("bottle.id", "test.id")], by = "bottle.id")

res <- list()
for (tid in unique(bottles$test.id)) {
  st <- bottles[bottles$test.id == tid, ]
  if (!any(st$substrate == "BK")) next
  vl <- gas[gas$test.id == tid, c("bottle.id", "time.d", "cvCH4")]
  st$bottle.id <- as.character(st$bottle.id); vl$bottle.id <- as.character(vl$bottle.id)
  r <- tryCatch(
    summBg(vol = vl, setup = st, id.name = "bottle.id", time.name = "time.d",
           descrip.name = "cond", inoc.name = "BK", inoc.m.name = "m.inoc",
           norm.name = "m.sub.vs", vol.name = "cvCH4", when = "1p3d",
           show.obs = TRUE, quiet = TRUE),
    error = function(e) { message("test ", tid, ": ", conditionMessage(e)); NULL })
  if (!is.null(r)) { r$test.id <- tid; res[[length(res) + 1]] <- r }
}
obs <- do.call(rbind, res)
write.csv(obs, file.path(out, "iis_ystar_biogas_bottles.csv"), row.names = FALSE)
cat("bottle rows:", nrow(obs), "\n"); print(head(obs))
