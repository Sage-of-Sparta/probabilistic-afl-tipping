
# libraries
library(dplyr)
library(fitzRoy)

set_lists_to_chars <- function(x) {
  if(class(x) == 'list') {
    y <- paste(unlist(x[1]), sep='', collapse=', ')
  } else {
    y <- x 
  }
  return(y)
}

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# get historical afl match results
results_2019_temp <- fitzRoy::fetch_results_afl(season = 2019, round_number = NULL, comp = "AFLM")
results_2019 <- data.frame(lapply(results_2019_temp, set_lists_to_chars), stringsAsFactors = F)
write.csv(results_2019,"data//afl_results_2019.csv", row.names = FALSE)

results_2020_temp <- fitzRoy::fetch_results_afl(season = 2020, round_number = NULL, comp = "AFLM")
results_2020 <- data.frame(lapply(results_2020_temp, set_lists_to_chars), stringsAsFactors = F)
write.csv(results_2020,"data//afl_results_2020.csv", row.names = FALSE)

results_2021_temp <- fitzRoy::fetch_results_afl(season = 2021, round_number = NULL, comp = "AFLM")
results_2021 <- data.frame(lapply(results_2021_temp, set_lists_to_chars), stringsAsFactors = F)
write.csv(results_2021,"data//afl_results_2021.csv", row.names = FALSE)

results_2022_temp <- fitzRoy::fetch_results_afl(season = 2022, round_number = NULL, comp = "AFLM")
results_2022 <- data.frame(lapply(results_2022_temp, set_lists_to_chars), stringsAsFactors = F)
write.csv(results_2022,"data//afl_results_2022.csv", row.names = FALSE)

fixture_2022_temp <- fitzRoy::fetch_fixture_afl(season = 2022, round_number = NULL, comp = "AFLM")
fixture_2022 <- data.frame(lapply(fixture_2022_temp, set_lists_to_chars), stringsAsFactors = F)
write.csv(fixture_2022,"data//fixture_2022.csv", row.names = FALSE)


# afltables_stats_temp <- fitzRoy::fetch_player_stats_afltables(season = 2021, round_number = NULL)
# afltables_stats <- data.frame(lapply(afltables_stats_temp, set_lists_to_chars), stringsAsFactors = F)
# write.csv(afltables_stats,"data//afltables_stats_2021.csv", row.names = FALSE)


