library(broom)
library(dplyr)
library(emmeans)

options(contrasts = c("contr.sum", "contr.poly"))

data_path <- '../data'
runid <- '20200226_connectivity'
fname <- sprintf('%s/all_conn_%s_for_stats.csv', data_path, runid)

conn <- read.csv(fname, sep=';')


conn <- conn %>% filter(SO %in% c('Group1', 'Group2', 'Group3', 'Group4'))
conn$SO <- factor(conn$SO)

lma <- lm(
    nice_marker_SymbolicMutualInformation_alpha_weighted ~ SO + MR + SO:MR + SO:MR:Connection,
    data=conn)

anovaa <- anova(lma)
    

lmt <- lm(
    nice_marker_SymbolicMutualInformation_theta_weighted ~ SO + MR + SO:MR + SO:MR:Connection,
    data=conn)

anovat <- anova(lmt)

message('==========')
message('WSMI Alpha')
message('==========')
print(anovaa)

message('==========')
message('WSMI Theta')
message('==========')
print(anovat)


emm_a <- emmeans(lma, pairwise ~ MR | SO * Connection, nesting=NULL)
emm_t <- emmeans(lmt, pairwise ~ MR | SO * Connection, nesting=NULL)

dfwsmia <- as.data.frame(emm_a$contrasts)
dfwsmit <- as.data.frame(emm_t$contrasts)

write.csv2(dfwsmia, '../stats/stats_conn_posthoc_wsmialpha.csv')
write.csv2(dfwsmit, '../stats/stats_conn_posthoc_wsmitheta.csv')