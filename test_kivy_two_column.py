import kivy
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.popup import Popup

kivy.config.Config.set('graphics', 'width', '400')
kivy.config.Config.set('graphics', 'height', '200')


class TwoColumnGUI(GridLayout):
    def __init__(self, **kwargs):
        super(TwoColumnGUI, self).__init__(**kwargs)

        self.cols = 2
        self.spacing = [10, 10]

        self.label1 = Label(text="Label:")
        self.add_widget(self.label1)

        self.text_input = TextInput(multiline=False)
        self.add_widget(self.text_input)

        self.button = Button(text="Say Hello")
        self.button.bind(on_press=self.on_button_press)
        self.add_widget(self.button)

    def on_button_press(self, instance):
        message = self.text_input.text
        popup = Popup(title='Hello Dialog', content=Label(text=f'Hello {message}!'), size_hint=(None, None), size=(400, 200))
        popup.open()


class TwoColumnGUIApp(App):
    def build(self):
        return TwoColumnGUI()


if __name__ == '__main__':
    TwoColumnGUIApp().run()
# Write your code here :-)
