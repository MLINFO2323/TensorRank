const template = document.createElement("template")
template.innerHTML = `
    <link rel="stylesheet" href="/components/collapsableDiv.css"/>
    <button type="button" class="collapsable">Use titile attr to change this</button>
    <div class="content">
        <slot></slot>
    </div>
`

class CollapsibleDiv extends HTMLElement {
    static get observedAttributes() {
        return ['title']
    }

    constructor() {
        super()
        const shadow = this.attachShadow({ mode: "open" })
        shadow.append(template.content.cloneNode(true))
        let titleBut = this.shadowRoot.querySelector("button")
        this.titleBut = titleBut;
        let contentDiv = this.shadowRoot.querySelector("div")
        this.contentDiv = contentDiv;
        titleBut.addEventListener("click", function () {
            titleBut.classList.toggle("active")
            if (contentDiv.style.maxHeight) {
                contentDiv.style.maxHeight = null;
            }
            else {
                contentDiv.style.maxHeight = contentDiv.scrollHeight + "px"
            }
        })
    }

    attributeChangedCallback(attrName, oldVal, newVal) {
        if (attrName == "title") {
            this.titleBut.innerHTML = newVal;
        }
    }
}
customElements.define("collapsable-div", CollapsibleDiv)