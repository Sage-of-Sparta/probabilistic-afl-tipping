
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


#results_2012_temp <- fitzRoy::fetch_results_afl(season = 2012, round_number = NULL, comp = "AFLM")
#results_2012 <- data.frame(lapply(results_2012_temp, set_lists_to_chars), stringsAsFactors = F)
#write.csv(results_2012,"data//afl_results_2012.csv", row.names = FALSE)

#results_2013_temp <- fitzRoy::fetch_results_afl(season = 2013, round_number = NULL, comp = "AFLM")
#results_2013 <- data.frame(lapply(results_2013_temp, set_lists_to_chars), stringsAsFactors = F)
#write.csv(results_2013,"data//afl_results_2013.csv", row.names = FALSE)

#results_2014_temp <- fitzRoy::fetch_results_afl(season = 2014, round_number = NULL, comp = "AFLM")
#results_2014 <- data.frame(lapply(results_2014_temp, set_lists_to_chars), stringsAsFactors = F)
#write.csv(results_2014,"data//afl_results_2014.csv", row.names = FALSE)

#results_2015_temp <- fitzRoy::fetch_results_afl(season = 2015, round_number = NULL, comp = "AFLM")
#results_2015 <- data.frame(lapply(results_2015_temp, set_lists_to_chars), stringsAsFactors = F)
#write.csv(results_2015,"data//afl_results_2015.csv", row.names = FALSE)

#results_2016_temp <- fitzRoy::fetch_results_afl(season = 2016, round_number = NULL, comp = "AFLM")
#results_2016 <- data.frame(lapply(results_2016_temp, set_lists_to_chars), stringsAsFactors = F)
#write.csv(results_2016,"data//afl_results_2016.csv", row.names = FALSE)

#results_2017_temp <- fitzRoy::fetch_results_afl(season = 2017, round_number = NULL, comp = "AFLM")
#results_2017 <- data.frame(lapply(results_2017_temp, set_lists_to_chars), stringsAsFactors = F)
#write.csv(results_2017,"data//afl_results_2017.csv", row.names = FALSE)

#results_2018_temp <- fitzRoy::fetch_results_afl(season = 2018, round_number = NULL, comp = "AFLM")
#results_2018 <- data.frame(lapply(results_2018_temp, set_lists_to_chars), stringsAsFactors = F)
#write.csv(results_2018,"data//afl_results_2018.csv", row.names = FALSE)

#results_2019_temp <- fitzRoy::fetch_results_afl(season = 2019, round_number = NULL, comp = "AFLM")
#results_2019 <- data.frame(lapply(results_2019_temp, set_lists_to_chars), stringsAsFactors = F)
#write.csv(results_2019,"data//afl_results_2019.csv", row.names = FALSE)

#results_2020_temp <- fitzRoy::fetch_results_afl(season = 2020, round_number = NULL, comp = "AFLM")
#results_2020 <- data.frame(lapply(results_2020_temp, set_lists_to_chars), stringsAsFactors = F)
#write.csv(results_2020,"data//afl_results_2020.csv", row.names = FALSE)

#results_2021_temp <- fitzRoy::fetch_results_afl(season = 2021, round_number = NULL, comp = "AFLM")
#results_2021 <- data.frame(lapply(results_2021_temp, set_lists_to_chars), stringsAsFactors = F)
#write.csv(results_2021,"data//afl_results_2021.csv", row.names = FALSE)

results_2022_temp <- fitzRoy::fetch_results_afl(season = 2022, round_number = NULL, comp = "AFLM")
results_2022 <- data.frame(lapply(results_2022_temp, set_lists_to_chars), stringsAsFactors = F)
write.csv(results_2022,"data//afl_results_2022.csv", row.names = FALSE)


fixture_2022_temp <- fitzRoy::fetch_fixture_afl(season = 2022, round_number = NULL, comp = "AFLM")
fixture_2022 <- data.frame(lapply(fixture_2022_temp, set_lists_to_chars), stringsAsFactors = F)
write.csv(fixture_2022,"data//fixture_2022.csv", row.names = FALSE)

afltables_stats_temp <- fitzRoy::fetch_player_stats_afltables(season = 2022, round_number = NULL)
afltables_stats <- data.frame(lapply(afltables_stats_temp, set_lists_to_chars), stringsAsFactors = F)
write.csv(afltables_stats,"data//afl_players_stats_2022.csv", row.names = FALSE)

#afltables_stats_temp <- fitzRoy::fetch_player_stats_afltables(season = 2021, round_number = NULL)
#afltables_stats <- data.frame(lapply(afltables_stats_temp, set_lists_to_chars), stringsAsFactors = F)
#write.csv(afltables_stats,"data//afl_players_stats_2021.csv", row.names = FALSE)

#afltables_stats_temp <- fitzRoy::fetch_player_stats_afltables(season = 2020, round_number = NULL)
#afltables_stats <- data.frame(lapply(afltables_stats_temp, set_lists_to_chars), stringsAsFactors = F)
#write.csv(afltables_stats,"data//afl_players_stats_2020.csv", row.names = FALSE)

#afltables_stats_temp <- fitzRoy::fetch_player_stats_afltables(season = 2019, round_number = NULL)
#afltables_stats <- data.frame(lapply(afltables_stats_temp, set_lists_to_chars), stringsAsFactors = F)
#write.csv(afltables_stats,"data//afl_players_stats_2019.csv", row.names = FALSE)


#afltables_stats_temp <- fitzRoy::fetch_player_stats_afltables(season = 2018, round_number = NULL)
#afltables_stats <- data.frame(lapply(afltables_stats_temp, set_lists_to_chars), stringsAsFactors = F)
#write.csv(afltables_stats,"data//afl_players_stats_2018.csv", row.names = FALSE)

#afltables_stats_temp <- fitzRoy::fetch_player_stats_afltables(season = 2017, round_number = NULL)
#afltables_stats <- data.frame(lapply(afltables_stats_temp, set_lists_to_chars), stringsAsFactors = F)
#write.csv(afltables_stats,"data//afl_players_stats_2017.csv", row.names = FALSE)


aflt_ladder_temp <- fitzRoy::fetch_ladder(season = 2022,round_number = NULL,comp = "AFLM",source = "AFL")
aflt_ladder <- data.frame(lapply(aflt_ladder_temp, set_lists_to_chars), stringsAsFactors = F)
write.csv(aflt_ladder,"data//afl_ladder_2022.csv", row.names = FALSE)




