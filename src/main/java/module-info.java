module org.mouse.mousecameraapp {
    requires javafx.controls;
    requires javafx.fxml;

    requires org.controlsfx.controls;
    requires org.kordamp.bootstrapfx.core;

    opens org.mouse.mousecameraapp to javafx.fxml;
    exports org.mouse.mousecameraapp;
}