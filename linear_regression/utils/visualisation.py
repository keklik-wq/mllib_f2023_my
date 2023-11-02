import plotly.graph_objects as go

#functions = dict([("polynom", 0), ("sin", 1), ("cos", 2), ("exp", 3)])
color_map = dict([("polynom", 'blue'), ("sin", 'red'), ("cos", 'yellow'), ("exp", 'black'),("log",'green')])


class Visualisation():

    def compare_model_predictions(self, x_values, predictions, y_actual, title):
        big_fig = go.Figure()
        big_fig.add_trace(go.Scatter(x=x_values, y=y_actual, mode='markers', name="data"))
        y_values = [row[0] for row in predictions]
        for iter in range(len(y_values)):
            sorted_x, sorted_y = zip(*sorted(zip(x_values, y_values[iter])))
            big_fig.add_trace(
                go.Scatter(x=sorted_x,
                           y=sorted_y,
                           mode='lines',
                           name=f"prediction_{iter}:{predictions[iter][1]}",
                           line=dict(color = color_map[predictions[iter][1]["function"]],
                                           width = 2)
                           )
            )
        big_fig.show()
