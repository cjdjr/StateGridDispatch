import example
grid = example.Print()
load_p_filepath = 'data/load_p.csv'
load_q_filepath = 'data/load_q.csv'
gen_p_filepath = 'data/gen_p.csv'
gen_q_filepath = 'data/gen_q.csv'

grid.readdata(1, load_p_filepath, load_q_filepath, gen_p_filepath, gen_q_filepath)
grid.env_feedback(grid.name_unp, grid.itime_unp[0], [], 1, [])
index = grid.un_nameindex
grid.readdata(1, load_p_filepath, load_q_filepath, gen_p_filepath, gen_q_filepath)
grid.env_feedback(index, grid.itime_unp[0], [], 1, [])
print(grid.index)
