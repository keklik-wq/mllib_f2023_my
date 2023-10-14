import plotly.graph_objects as go

functions = dict([("polynom", 0), ("sin", 1), ("cos", 2), ("exp", 3)])


class Visualisation():

    def compare_model_predictions(self, x_values, predictions, y_actual, title):
        big_fig = go.Figure()
        big_fig.add_trace(go.Scatter(x=x_values, y=y_actual, mode='markers', name="data"))
        y_values = [row[0] for row in predictions]
        for iter in range(len(y_values)):
            sorted_x, sorted_y = zip(*sorted(zip(x_values, y_values[iter])))
            big_fig.add_trace(
                go.Scatter(x=sorted_x, y=sorted_y, mode='lines', name=f"prediction_{iter}:{predictions[iter][1]}"))
        big_fig.show()
