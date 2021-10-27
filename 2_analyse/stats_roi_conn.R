library(broom)
library(dplyr)
library(emmeans)

options(contrasts = c("contr.sum", "contr.poly"))

data_path <- '../data'
runid <- '09092021_connectivity'
fname <- sprintf('%s/all_conn_%s_for_stats.csv', data_path, runid)

conn <- read.csv(fname, sep=';')


conn <- conn %>% filter(SO %in% c('H1', 'H2', 'H3', 'H4', 'H5', 'H6to8'))
conn$SO <- factor(conn$SO)

lma <- lm(
    nice_marker_SymbolicMutualInformation_alpha_weighted ~ SO + MR + SO:MR + SO:MR:Connection,
    data=conn)

anovaa <- anova(lma)
    

lmt <- lm(
    nice_marker_SymbolicMutualInformation_theta_weighted ~ SO + MR + SO:MR + SO:MR:Connection,
    data=conn)

anovat <- anova(lmt)

lmr <- lm(
    nice_sandbox_marker_Ratio_wsmi_theta_alpha ~ SO + MR + SO:MR + SO:MR:Connection,
    data=conn)

anovar <- anova(lmr)

message('==========')
message('WSMI Alpha')
message('==========')
print(anovaa)

message('==========')
message('WSMI Theta')
message('==========')
print(anovat)

message('================')
message('WSMI Theta/Alpha')
message('================')
print(anovar)



emm_a <- emmeans(lma, pairwise ~ MR | SO * Connection, nesting=NULL)
emm_t <- emmeans(lmt, pairwise ~ MR | SO * Connection, nesting=NULL)
emm_r <- emmeans(lmr, pairwise ~ MR | SO * Connection, nesting=NULL)

dfwsmia <- as.data.frame(emm_a$contrasts)
dfwsmit <- as.data.frame(emm_t$contrasts)
dfwsmir <- as.data.frame(emm_r$contrasts)

write.csv2(dfwsmia, '../stats/stats_conn_posthoc_wsmialpha.csv')
write.csv2(dfwsmit, '../stats/stats_conn_posthoc_wsmitheta.csv')
write.csv2(dfwsmir, '../stats/stats_conn_posthoc_wsmithetaalpha.csv')