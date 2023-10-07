import plotly.graph_objects as go


class Visualisation():

    def compare_model_predictions(self, x_values, y_values_list, y_actual, title):
        big_fig = go.Figure()
        big_fig.add_trace(go.Scatter(x=x_values, y=y_actual, mode='markers', name="data"))
        for iter in range(len(y_values_list)):
            sorted_x, sorted_y = zip(*sorted(zip(x_values, y_values_list[iter])))
            big_fig.add_trace(go.Scatter(x=sorted_x, y=sorted_y, mode='lines', name=f"prediction_{iter}"))
        big_fig.show()
