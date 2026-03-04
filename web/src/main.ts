import { app } from "../../scripts/app.js"
import { createApp, h } from "vue"
import EaseCurveWidget from "./components/EaseCurveWidget.vue"

const EASING_NODE_CLASSES = ["EaseCurve", "ApplyEasingToFloats"]

app.registerExtension({
  name: "nodesweet.easeCurve",

  nodeCreated(node: any) {
    if (!EASING_NODE_CLASSES.includes(node.comfyClass)) return

    const presetWidget = node.widgets?.find((w: any) => w.name === "preset")
    if (!presetWidget) return

    // Create container for the Vue curve preview
    const container = document.createElement("div")
    container.style.width = "100%"

    // Mount Vue app
    const vueApp = createApp({
      render() {
        return h(EaseCurveWidget, {
          node,
          presetWidget,
        })
      },
    })
    vueApp.mount(container)

    // Add as a DOM widget in ComfyUI
    const widget = node.addDOMWidget("curve_preview", "div", container, {
      serialize: false,
      hideOnZoom: false,
    })
    widget.computeSize = () => [200, 260]

    // Set a reasonable minimum node size
    node.setSize([320, node.computeSize()[1]])
  },
})
