# From https://gist.github.com/MattJBritton/9dc26109acb4dfe17820cf72d82f1e6f
import ipywidgets as wid
from IPython.display import display


def multi_checkbox_widget(options_dict):
    """ Widget with a search field and lots of checkboxes """
    search_widget = wid.Text(width="auto", layout=wid.Layout(flex="1 1 auto", width="auto", overflow='hidden'))
    output_widget = wid.Output()
    options = [x for x in options_dict.values()]
    options_layout = wid.Layout(
        overflow='auto',
        border='1px solid black',
        width='170px',
        height='440px',
        flex_flow='column',
        display='flex'
    )

    options_widget = wid.VBox(options, layout=options_layout)
    multi_select = wid.VBox([search_widget, options_widget])

    # Wire the search field to the checkboxes
    @output_widget.capture()
    def on_text_change(change):
        search_input = change['new']
        if search_input == '':
            # Reset search field
            new_options = sorted(options, key=lambda x: x.value, reverse=True)
        else:
            # Filter by search field using difflib.
            # close_matches = difflib.get_close_matches(search_input, list(options_dict.keys()), cutoff=0.0)
            close_matches = [x for x in list(options_dict.keys()) if str.lower(search_input.strip('')) in str.lower(x)]
            new_options = sorted(
                [x for x in options if x.description in close_matches],
                key=lambda x: x.value, reverse=True
            )  # [options_dict[x] for x in close_matches]
        options_widget.children = new_options

    search_widget.observe(on_text_change, names='value')
    display(output_widget)
    return multi_select
