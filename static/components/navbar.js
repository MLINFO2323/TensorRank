class Navbar extends HTMLElement {
    constructor() {
        super()
    }
    connectedCallback() {

        this.innerHTML = `

        <div
            style="background: #2b2b2b;display: flex;flex-direction: row;padding:0px;height: 120px;justify-content: space-between;">
            <hamburger-menu></hamburger-menu>
            <div style="display: flex;flex-direction: row;align-items: center;">
                <h1>TensorRank</h1>
                <img src="/assets/tf.svg" height="70px" style="margin:0px 20px" alt="" />
            </div>
        </div>
        `
    }
}

customElements.define("nav-bar", Navbar)